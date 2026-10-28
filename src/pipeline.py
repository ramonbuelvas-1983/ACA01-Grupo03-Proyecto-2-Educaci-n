"""TRAYECTORIA-UCI: recuperación documental y perfiles descriptivos de permanencia."""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import io
import json
import os
from pathlib import Path
import re
import sqlite3
import time
import unicodedata
from urllib.parse import urlparse
import urllib.request
import uuid
import zipfile

import fitz
import numpy as np
import pandas as pd
from dotenv import load_dotenv

MODEL = "intfloat/multilingual-e5-small"
REVISION = "614241f622f53c4eeff9890bdc4f31cfecc418b3"
MAX_TOKENS = 280
MAX_PARAGRAPH_OVERLAP = 50
LONG_PARAGRAPH_OVERLAP = 40
RERANK_MODEL = "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1"
RERANK_REVISION = "1427fd652930e4ba29e8149678df786c240d8825"
CANDIDATE_POOL = 20
PAIR_TOKEN_LIMIT = 512


def file_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def object_hash(value):
    encoded = json.dumps(value, ensure_ascii=False, sort_keys=True).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def fold(text):
    value = unicodedata.normalize("NFKD", text).casefold()
    return " ".join("".join(c for c in value if not unicodedata.combining(c)).split())


def theme_for(text):
    normalized = fold(text)
    lexicon = {
        "academic": ["academ", "tutori", "curricul", "aprendiz", "rendimiento", "nivelacion"],
        "financial": ["financ", "econom", "beca", "deud", "credito educativo", "subsid"],
        "management": ["gestion", "institucion", "indicador", "seguimiento", "alerta", "spadies"],
    }
    scores = {key: sum(normalized.count(word) for word in words) for key, words in lexicon.items()}
    return max(scores, key=scores.get) if max(scores.values()) else "management"



INSTITUTION_ALIASES = {
    "ministerio de educacion nacional": ["MEN", "Ministerio de Educación Nacional"],
    "universidad el bosque": ["Universidad El Bosque", "El Bosque"],
    "universidad nacional de colombia": ["UNAL", "Universidad Nacional de Colombia"],
}


def detect_source_scope(query,documents):
    """Resuelve una mención institucional explícita sin consultar el banco de evaluación."""
    normalized=fold(query)
    matches={}
    for document in documents:
        identity=fold(str(document.get("institution",document.get("author",""))))
        aliases=[]
        for institution,names in INSTITUTION_ALIASES.items():
            if institution in identity:
                aliases.extend(names)
        found=[alias for alias in aliases
               if re.search(r"(?<!\w)"+re.escape(fold(alias))+r"(?!\w)",normalized)]
        if found:
            matches[document["id"]]=found
    ids=sorted(matches)
    mode="explicit_single_source" if len(ids)==1 else ("multiple_sources" if ids else "all_sources")
    return {"mode":mode,"document_id":ids[0] if len(ids)==1 else None,
            "matched_source_ids":ids,"matched_aliases":matches,
            "policy":"solo_una_fuente_explicita_activa_filtro; cero_o_varias_consultan_todo_el_corpus"}


def paragraph_units(paragraphs, tokenizer):
    """Crea unidades con offsets exactos en una transcripción normalizada por página."""
    page_text = "\n\n".join(paragraphs)
    units = []
    cursor = 0
    for paragraph in paragraphs:
        ids = tokenizer(paragraph, add_special_tokens=False, return_offsets_mapping=True,
                        truncation=False, verbose=False)
        offsets = ids["offset_mapping"]
        if len(offsets) <= MAX_TOKENS:
            if paragraph:
                units.append({"start": cursor, "end": cursor+len(paragraph),
                              "tokens": len(offsets), "whole_paragraph": True})
        else:
            left = 0
            while left < len(offsets):
                right = min(left+MAX_TOKENS, len(offsets))
                start, end = offsets[left][0], offsets[right-1][1]
                piece = paragraph[start:end]
                # Al cortar subpalabras, el recuento de una nueva tokenización puede variar.
                while len(tokenizer(piece, add_special_tokens=False)["input_ids"]) > MAX_TOKENS:
                    right -= 1
                    end = offsets[right-1][1]
                    piece = paragraph[start:end]
                units.append({"start": cursor+start, "end": cursor+end,
                              "tokens":len(tokenizer(piece,add_special_tokens=False)["input_ids"]),
                              "whole_paragraph": False})
                if right == len(offsets):
                    break
                left = max(left+1,right-LONG_PARAGRAPH_OVERLAP)
        cursor += len(paragraph)+2
    return page_text, units


def compose_chunks(paragraphs, tokenizer):
    """Acumula párrafos completos; conserva el último si tiene como máximo 50 tokens."""
    page_text, units = paragraph_units(paragraphs, tokenizer)
    pending = []
    emitted = set()

    def package(group):
        a,b = group[0]["start"],group[-1]["end"]
        text = page_text[a:b]
        tokens = len(tokenizer(text,add_special_tokens=False)["input_ids"])
        if tokens > MAX_TOKENS:
            raise ValueError("Fragmento mayor que el presupuesto de tokens.")
        return {"text":text,"token_count":tokens,"char_start":a,"char_end":b}

    for unit in units:
        if pending:
            combined = page_text[pending[0]["start"]:unit["end"]]
            count = len(tokenizer(combined,add_special_tokens=False)["input_ids"])
            if count > MAX_TOKENS:
                chunk=package(pending)
                if (chunk["char_start"],chunk["char_end"]) not in emitted:
                    emitted.add((chunk["char_start"],chunk["char_end"]))
                    yield chunk
                tail=pending[-1]
                pending=[tail] if tail["whole_paragraph"] and tail["tokens"] <= MAX_PARAGRAPH_OVERLAP else []
                if pending:
                    merged=page_text[pending[0]["start"]:unit["end"]]
                    if len(tokenizer(merged,add_special_tokens=False)["input_ids"]) > MAX_TOKENS:
                        pending=[]
        pending.append(unit)
    if pending:
        chunk=package(pending)
        if (chunk["char_start"],chunk["char_end"]) not in emitted:
            yield chunk


class TextEncoder:
    """Un espacio vectorial por proveedor; la colección incorpora su identidad."""
    def __init__(self,provider):
        from transformers import AutoTokenizer
        if provider not in {"e5","gemini2"}:
            raise ValueError("Proveedor válido: e5 o gemini2.")
        self.provider=provider
        self.name=MODEL if provider=="e5" else "gemini-embedding-2"
        self.dimension=384 if provider=="e5" else 768
        self.tokenizer=AutoTokenizer.from_pretrained(MODEL,revision=REVISION)
        self.model=None
        self.api=None
        if provider=="gemini2":
            key=os.environ.get("GEMINI_API_KEY")
            if not key:
                raise RuntimeError("Defina GEMINI_API_KEY en el entorno o en .env privado.")
            from google import genai
            self.api=genai.Client(api_key=key)

    def transform(self,texts,role,titles=None):
        if role not in {"RETRIEVAL_DOCUMENT","RETRIEVAL_QUERY"}:
            raise ValueError("El rol debe ser documental o de consulta.")
        if self.provider=="e5":
            if self.model is None:
                import torch
                from sentence_transformers import SentenceTransformer
                torch.set_num_threads(min(4,os.cpu_count() or 1))
                self.model=SentenceTransformer(MODEL,revision=REVISION,device="cpu")
                self.model.max_seq_length=512
            prefix="query: " if role=="RETRIEVAL_QUERY" else "passage: "
            prepared=[prefix+text for text in texts]
            if any(len(self.tokenizer(t,add_special_tokens=True)["input_ids"])>512 for t in prepared):
                raise ValueError("Entrada superior a 512 tokens: el pipeline impide truncamiento silencioso.")
            matrix=self.model.encode(prepared,batch_size=24,convert_to_numpy=True,
                                     normalize_embeddings=True,show_progress_bar=False)
        else:
            from google.genai import types
            values=[]
            for number,text in enumerate(texts):
                # Gemini 2 traduce los roles a instrucciones; no admite task_type.
                prepared=(f"task: search result | query: {text}" if role=="RETRIEVAL_QUERY"
                          else f"title: {titles[number] if titles else 'none'} | text: {text}")
                for attempt in range(5):
                    try:
                        response=self.api.models.embed_content(model=self.name,contents=prepared,
                                      config=types.EmbedContentConfig(output_dimensionality=self.dimension))
                        if len(response.embeddings or [])!=1:
                            raise ValueError("Se esperaba un vector por entrada.")
                        values.append(response.embeddings[0].values)
                        break
                    except Exception as error:
                        code=getattr(error,"code",None)
                        if code not in {429,500,502,503,504} or attempt==4:
                            raise RuntimeError("El proveedor de embeddings no completó la solicitud; revise cuota y credenciales en privado.") from None
                        time.sleep(2**attempt)
                time.sleep(float(os.getenv("GEMINI_REQUEST_INTERVAL","1.0")))
            matrix=np.asarray(values,dtype=np.float32)
        matrix=np.asarray(matrix,dtype=np.float32)
        if matrix.shape!=(len(texts),self.dimension) or not np.isfinite(matrix).all():
            raise ValueError("La forma o los valores del embedding son inválidos.")
        norm=np.linalg.norm(matrix,axis=1,keepdims=True)
        if np.any(norm<=0):
            raise ValueError("No se aceptan vectores nulos.")
        return matrix/norm


def ranked_candidates(candidates, logits, k):
    """Ordena por logit del cross-encoder conservando intacto el coseno denso."""
    values=np.asarray(logits).reshape(-1)
    if len(values)!=len(candidates) or not np.isfinite(values).all():
        raise ValueError("El reranker debe devolver una puntuación finita por candidato.")
    scored=[dict(row,rerank_score=float(value)) for row,value in zip(candidates,values)]
    ordered=sorted(scored,key=lambda row:(-row["rerank_score"],row["dense_rank"]))
    return [dict(row,rank=rank) for rank,row in enumerate(ordered[:k],1)],scored



def ensure_source_payload_index(client,collection):
    """Verifica el índice keyword exigido por filtros de fuente en cloud estricto."""
    from qdrant_client import models
    schema=client.get_collection(collection).payload_schema or {}
    existing=schema.get("doc_id")
    created=existing is None
    if existing is not None and existing.data_type!=models.PayloadSchemaType.KEYWORD:
        raise ValueError("El índice payload doc_id existente debe ser de tipo keyword.")
    if created:
        client.create_payload_index(collection_name=collection,field_name="doc_id",
                                    field_schema=models.PayloadSchemaType.KEYWORD,wait=True)
    verified_schema=client.get_collection(collection).payload_schema or {}
    verified=verified_schema.get("doc_id")
    if verified is None or verified.data_type!=models.PayloadSchemaType.KEYWORD:
        raise ValueError("No se confirmó el índice keyword de doc_id en el clúster remoto.")
    return {"field":"doc_id","schema":"keyword","verified":True,"created":created,
            "creation_waited":created,"verified_before_upsert":True}


class StudentEvidencePipeline:
    def __init__(self,root=".",backend="local",provider="e5",persistent=True,retrieval="rerank"):
        self.root=Path(root).resolve()
        load_dotenv(self.root/".env",override=False)
        if backend not in {"local","cloud"}:
            raise ValueError("Seleccione backend local o cloud explícitamente.")
        if retrieval not in {"dense","rerank"}:
            raise ValueError("Seleccione recuperación dense o rerank.")
        self.backend,self.provider,self.persistent=backend,provider,persistent
        self.retrieval=retrieval
        self.reranker=None
        self.evaluation_runs={}
        self.dataset=json.loads((self.root/"data/dataset.json").read_text())
        self.output=self.root/"results"
        self.output.mkdir(parents=True,exist_ok=True)
        self.cache=self.root/".trajectory_cache"
        self.cache.mkdir(exist_ok=True)
        self.encoder=None
        self.client=None
        self.vectors=None
        self.chunks=[]
        self.db_path=self.output/"cohorts.sqlite"

    def _restore_dataset(self):
        path=self.root/self.dataset["csv_path"]
        if path.exists():
            return path
        with urllib.request.urlopen(self.dataset["download_url"],timeout=60) as remote:
            compressed=remote.read()
        with zipfile.ZipFile(io.BytesIO(compressed)) as archive:
            candidates=[name for name in archive.namelist() if Path(name).name=="data.csv"]
            if len(candidates)!=1:
                raise ValueError("No se identificó inequívocamente el CSV original.")
            content=archive.read(candidates[0])
        if hashlib.sha256(content).hexdigest()!=self.dataset["csv_sha256"]:
            raise ValueError("El dataset descargado difiere del archivo fuente registrado.")
        path.parent.mkdir(parents=True,exist_ok=True)
        path.write_bytes(content)
        return path

    def ingest(self):
        path=self._restore_dataset()
        if file_hash(path)!=self.dataset["csv_sha256"]:
            raise ValueError("El CSV no coincide con la huella del archivo suministrado.")
        records=pd.read_csv(path,sep=self.dataset["delimiter"],encoding=self.dataset["encoding"])
        records.columns=records.columns.str.strip()
        if len(records)!=4424 or len(records.columns)!=37:
            raise ValueError("La forma del dataset no coincide con UCI 697.")
        counts=records["Target"].value_counts()
        self.dataset_summary={"rows":len(records),"features":len(records.columns)-1,
             "missing_cells":int(records.isna().sum().sum()),"duplicate_rows":int(records.duplicated().sum()),
             "class_counts":{str(k):int(v) for k,v in counts.items()},"country":"Portugal",
             "sha256":self.dataset["csv_sha256"],"source":"archivo_suministrado_verificado"}
        approved=records["Curricular units 1st sem (approved)"]
        definitions=[("approval_0","academic","aprobadas_sem1","0",approved==0),
                     ("approval_1_5","academic","aprobadas_sem1","1-5",approved.between(1,5)),
                     ("approval_6_plus","academic","aprobadas_sem1","6+",approved>=6)]
        for value in [0,1]:
            definitions.append((f"scholarship_{value}","financial","beca",str(value),records["Scholarship holder"]==value))
            definitions.append((f"debtor_{value}","financial","deudor",str(value),records["Debtor"]==value))
        self.cohort_profiles=[]
        for ident,theme,grouping,label,mask in definitions:
            group=records.loc[mask,"Target"]
            n=len(group)
            self.cohort_profiles.append({"profile_id":ident,"thematic":theme,"grouping":grouping,
                "group_value":label,"n":n,"dropout_count":int(group.eq("Dropout").sum()),
                "graduate_count":int(group.eq("Graduate").sum()),"enrolled_count":int(group.eq("Enrolled").sum()),
                "dropout_rate":float(group.eq("Dropout").mean())})
        for grouping in {p["grouping"] for p in self.cohort_profiles}:
            assert sum(p["n"] for p in self.cohort_profiles if p["grouping"]==grouping)==4424
        with sqlite3.connect(self.db_path) as db:
            records.to_sql("student_records",db,index=False,if_exists="replace")
            pd.DataFrame(self.cohort_profiles).to_sql("cohort_profiles",db,index=False,if_exists="replace")
            assert db.execute("SELECT COUNT(*) FROM student_records").fetchone()[0]==4424
            assert db.execute("SELECT COUNT(*) FROM cohort_profiles").fetchone()[0]==7
        return self.dataset_summary

    def _load_documents(self):
        self.sources=json.loads((self.root/"data/sources.json").read_text())
        docs=self.sources["documents"]
        if len(docs)<3:
            raise ValueError("El corpus requiere al menos tres documentos complementarios.")
        for source in docs:
            path=self.root/source["relative_path"]
            if not path.exists():
                request=urllib.request.Request(source.get("download_url",source["url"]),headers={"User-Agent":"TRAYECTORIA-UCI/1.0"})
                with urllib.request.urlopen(request,timeout=60) as response:
                    content=response.read()
                if hashlib.sha256(content).hexdigest()!=source["sha256"]:
                    raise ValueError("La fuente pública cambió; revise su versión antes de actualizar.")
                path.parent.mkdir(parents=True,exist_ok=True)
                path.write_bytes(content)
            if file_hash(path)!=source["sha256"]:
                raise ValueError(f"No coincide la huella del documento {source['id']}.")
        return docs

    def prepare_corpus(self):
        if not hasattr(self,"cohort_profiles"):
            self.ingest()
        docs=self._load_documents()
        self.encoder=self.encoder or TextEncoder(self.provider)
        self.corpus_summary=[]
        self.chunks=[]
        page_audit=[]
        seen=set()
        for source in docs:
            with fitz.open(self.root/source["relative_path"]) as pdf:
                extracted=[]
                margins=[]
                for page in pdf:
                    blocks=[]
                    for block in page.get_text("blocks",sort=True):
                        if len(block)<7 or block[6]!=0:
                            continue
                        raw=unicodedata.normalize("NFC",block[4])
                        raw=re.sub(r"(?<=\w)-\n(?=\w)","",raw)
                        clean=" ".join(raw.split())
                        if clean and not re.fullmatch(r"\d+",clean):
                            blocks.append(clean)
                    extracted.append(blocks)
                    margins.extend(set(blocks[:2]+blocks[-2:]))
                frequency=Counter(margins)
                repeated={text for text,count in frequency.items() if len(text)<160 and count>=max(3,int(len(pdf)*.4))}
                for page_number,blocks in enumerate(extracted,1):
                    paragraphs=[text for text in blocks if text not in repeated]
                    character_count=sum(map(len,paragraphs))
                    page_audit.append({"doc_id":source["id"],"page":page_number,"characters":character_count})
                    if character_count<30:
                        continue
                    for chunk in compose_chunks(paragraphs,self.encoder.tokenizer):
                        text_hash=hashlib.sha256(chunk["text"].encode()).hexdigest()
                        duplicate=(source["id"],text_hash)
                        if duplicate in seen:
                            continue
                        seen.add(duplicate)
                        theme=theme_for(chunk["text"])
                        profile_ids=[p["profile_id"] for p in self.cohort_profiles
                                     if theme=="management" or p["thematic"]==theme]
                        ident=str(uuid.uuid5(uuid.NAMESPACE_DNS,
                              f"trajectory:{source['sha256']}:{page_number}:{chunk['char_start']}:{text_hash}"))
                        chunk.update(id=ident,doc_id=source["id"],page=page_number,title=source["title"],
                             url=source.get("download_url",source["url"]),original_url=source["url"],thematic=theme,profile_ids=profile_ids,text_sha256=text_hash,
                             source_sha256=source["sha256"],dataset_sha256=self.dataset["csv_sha256"],
                             integration="tematica_no_personal",dataset_country="Portugal")
                        self.chunks.append(chunk)
                self.corpus_summary.append({"doc_id":source["id"],"title":source["title"],
                                            "pages":len(pdf),"sha256":source["sha256"]})
        if not self.chunks:
            raise ValueError("El corpus no contiene texto legible: se requiere OCR verificable.")
        tokens=[c["token_count"] for c in self.chunks]
        self.chunk_stats={"count":len(tokens),"median_tokens":float(np.median(tokens)),
            "min_tokens":min(tokens),"max_tokens":max(tokens),"document_pages":len(page_audit),
            "pages_without_text":sum(p["characters"]<30 for p in page_audit),
            "overlap_policy":"ultimo_parrafo_completo_hasta_50_tokens; parrafo_largo_40_tokens",
            "max_chunk_tokens":MAX_TOKENS}
        self.embedding_manifest={"pipeline":"trajectory_uci_v1","model":self.encoder.name,
            "model_revision":REVISION if self.provider=="e5" else "managed_api", "tokenizer_revision":REVISION,
            "dimension":self.encoder.dimension,"max_tokens":MAX_TOKENS,
            "max_paragraph_overlap":MAX_PARAGRAPH_OVERLAP,"long_paragraph_overlap":LONG_PARAGRAPH_OVERLAP,
            "chunks":[c["id"] for c in self.chunks],"dataset_sha256":self.dataset["csv_sha256"]}
        self.fingerprint=object_hash(self.embedding_manifest)[:18]
        self.collection=f"trajectory_{self.provider}_{self.fingerprint}"
        (self.output/"page_extraction.json").write_text(json.dumps(page_audit,ensure_ascii=False,indent=2))
        return self.chunks

    def encode(self):
        if not self.chunks:
            self.prepare_corpus()
        array_path=self.cache/f"embeddings_{self.fingerprint}.npz"
        meta_path=self.cache/f"embeddings_{self.fingerprint}.json"
        self.cache_reused=False
        if array_path.exists() and meta_path.exists():
            if json.loads(meta_path.read_text())!=self.embedding_manifest:
                raise ValueError("El manifiesto de caché no coincide con las entradas.")
            with np.load(array_path,allow_pickle=False) as saved:
                self.vectors=saved["vectors"]
            self.cache_reused=True
        else:
            self.vectors=self.encoder.transform([c["text"] for c in self.chunks],"RETRIEVAL_DOCUMENT",
                                                titles=[c["title"] for c in self.chunks])
            np.savez_compressed(array_path,vectors=self.vectors)
            meta_path.write_text(json.dumps(self.embedding_manifest,ensure_ascii=False,indent=2))
        assert self.vectors.shape==(len(self.chunks),self.encoder.dimension)
        assert np.isfinite(self.vectors).all()
        assert np.allclose(np.linalg.norm(self.vectors,axis=1),1,atol=1e-5)
        return self.vectors

    def index(self):
        try:
            return self._index()
        except Exception as error:
            if self.backend=="cloud":
                if not os.getenv("QDRANT_URL") or not os.getenv("QDRANT_API_KEY"):
                    raise RuntimeError("Backend cloud requiere QDRANT_URL y QDRANT_API_KEY en el entorno; no se activa un sustituto local.") from None
                raise RuntimeError(f"No se verificó el clúster remoto ({type(error).__name__}); revise acceso, permisos y configuración sin publicar sus credenciales.") from None
            raise

    def _index(self):
        from qdrant_client import QdrantClient,models
        if self.vectors is None:
            self.encode()
        self.close()
        if self.backend=="cloud":
            address=os.environ.get("QDRANT_URL")
            secret=os.environ.get("QDRANT_API_KEY")
            if not address or not secret:
                raise ValueError("Credenciales ausentes.")
            parts=urlparse(address)
            if parts.scheme!="https" or not parts.hostname or parts.username or parts.hostname in {"localhost","127.0.0.1","0.0.0.0"}:
                raise ValueError("La URL debe ser HTTPS y corresponder a un clúster remoto.")
            self.client=QdrantClient(url=address,api_key=secret,timeout=60)
        elif self.persistent:
            self.client=QdrantClient(path=str(self.output/"qdrant_local"))
        else:
            self.client=QdrantClient(location=":memory:")
        if not self.client.collection_exists(self.collection):
            self.client.create_collection(self.collection,
                   vectors_config=models.VectorParams(size=self.encoder.dimension,distance=models.Distance.COSINE))
        settings=self.client.get_collection(self.collection).config.params.vectors
        if settings.size!=self.encoder.dimension or settings.distance!=models.Distance.COSINE:
            raise ValueError("La colección tiene un esquema vectorial incompatible.")
        payload_index_evidence=None
        if self.backend=="cloud":
            payload_index_evidence=ensure_source_payload_index(self.client,self.collection)
        def points(start,stop):
            return [models.PointStruct(id=self.chunks[i]["id"],vector=self.vectors[i].tolist(),payload=self.chunks[i])
                    for i in range(start,min(stop,len(self.chunks)))]
        for start in range(0,len(self.chunks),64):
            self.client.upsert(self.collection,points=points(start,start+64),wait=True)
        actual=self.client.count(self.collection,exact=True).count
        if actual!=len(self.chunks):
            raise ValueError("El número de puntos indexados no coincide con el corpus.")
        self.client.upsert(self.collection,points=points(0,64),wait=True)
        repeated=self.client.count(self.collection,exact=True).count
        assert repeated==actual
        probe=self.client.query_points(self.collection,query=self.vectors[0].tolist(),limit=1).points
        assert probe and probe[0].score>.999
        payload_bytes=sum(len(json.dumps(c,ensure_ascii=False).encode()) for c in self.chunks)
        self.index_report={"backend":self.backend,"remote_verified":self.backend=="cloud",
             "collection":self.collection,"dimension":self.encoder.dimension,"metric":"cosine",
             "expected_points":len(self.chunks),"stored_points":actual,"idempotent_count":repeated,
             "idempotency_test":"repeticion_del_primer_lote_con_UUID_estables", "batch_size":64,
             "vector_mib":self.vectors.nbytes/1024**2,"payload_mib":payload_bytes/1024**2,
             "raw_data_mib":(self.vectors.nbytes+payload_bytes)/1024**2,
             "capacity_scope":"datos_brutos_sin_indices_ni_sobrecarga_del_servidor",
             "self_match_score":float(probe[0].score),"verified_at":datetime.now(timezone.utc).isoformat()}
        if self.backend=="cloud":
            self.index_report["payload_index"]=payload_index_evidence
        return self.index_report

    def search(self,query,k=3,include_cohorts=True):
        if self.client is None:
            raise RuntimeError("Indexe el corpus antes de consultar.")
        if not isinstance(query,str) or not query.strip() or not 1<=k<=20:
            raise ValueError("Especifique una consulta no vacía y k entre 1 y 20.")
        vector=self.encoder.transform([query],"RETRIEVAL_QUERY")[0]
        candidate_limit=CANDIDATE_POOL if self.retrieval=="rerank" else k
        source_scope=detect_source_scope(query,self.sources["documents"])
        query_filter=None
        if source_scope["document_id"] is not None:
            from qdrant_client import models
            query_filter=models.Filter(must=[models.FieldCondition(key="doc_id",
                                      match=models.MatchValue(value=source_scope["document_id"]))])
        try:
            returned=self.client.query_points(self.collection,query=vector.tolist(),limit=candidate_limit,
                                              query_filter=query_filter,with_payload=True,with_vectors=True).points
        except Exception as error:
            if self.backend=="cloud":
                raise RuntimeError(f"Falló la consulta remota ({type(error).__name__}); no se sustituyó por una consulta local.") from None
            raise
        hits=[]
        for position,item in enumerate(returned,1):
            values=np.asarray(item.vector,dtype=np.float32)
            independent=float(vector@values/(np.linalg.norm(vector)*np.linalg.norm(values)))
            error=abs(independent-float(item.score))
            if error>1e-5:
                raise ValueError("El coseno recalculado difiere del valor del motor.")
            payload=item.payload
            hits.append({"rank":position,"dense_rank":position,"score":float(item.score),
                         "cosine_numpy":independent,"cosine_error":error,"chunk_id":payload["id"],
                         **{key:payload[key] for key in ["text","doc_id","page","title","url","thematic","profile_ids"]}})
        candidates=hits
        pair_max_tokens=None
        if self.retrieval=="rerank":
            if self.reranker is None:
                from sentence_transformers import CrossEncoder
                import torch
                torch.set_num_threads(min(4,os.cpu_count() or 1))
                self.reranker=CrossEncoder(RERANK_MODEL,revision=RERANK_REVISION,
                                           max_length=PAIR_TOKEN_LIMIT,device="cpu")
            pairs=[(query,f"Título: {hit['title']}\nPasaje: {hit['text']}") for hit in candidates]
            lengths=[len(self.reranker.tokenizer(question,passage,add_special_tokens=True,
                         truncation=False,verbose=False)["input_ids"]) for question,passage in pairs]
            pair_max_tokens=max(lengths,default=0)
            if pair_max_tokens>PAIR_TOKEN_LIMIT:
                raise ValueError(f"El par consulta-pasaje alcanza {pair_max_tokens} tokens y excede el límite de {PAIR_TOKEN_LIMIT}; acorte la consulta. No se truncó texto.")
            import torch
            logits=self.reranker.predict(pairs,batch_size=8,show_progress_bar=False,
                                        activation_fn=torch.nn.Identity(),convert_to_numpy=True)
            hits,candidates=ranked_candidates(candidates,logits,k)
        theme=theme_for(query)
        cohorts=[]
        # El puente se materializa en los IDs del payload recuperado; SQL aporta cifras exactas.
        linked_ids={ident for hit in hits for ident in hit["profile_ids"]}
        if theme!="management":
            allowed={p["profile_id"] for p in self.cohort_profiles if p["thematic"]==theme}
            linked_ids &= allowed
        if include_cohorts and linked_ids:
            ordered_ids=sorted(linked_ids)
            placeholders=",".join("?" for _ in ordered_ids)
            with sqlite3.connect(self.db_path) as db:
                db.row_factory=sqlite3.Row
                rows=db.execute(f"SELECT * FROM cohort_profiles WHERE profile_id IN ({placeholders}) ORDER BY profile_id",ordered_ids).fetchall()
                cohorts=[dict(row) for row in rows]
        trace_keys=["chunk_id","doc_id","page","dense_rank","score","cosine_error"]
        if self.retrieval=="rerank":
            trace_keys.append("rerank_score")
        return {"query":query,"backend":self.backend,"hits":hits,"cohorts":cohorts,"cohort_theme":theme,
                "ranking_method":self.retrieval,"candidate_count":len(candidates),"source_scope":source_scope,
                "reranker_pair_format":"query | Titulo: title + Pasaje: text" if self.retrieval=="rerank" else None,
                "candidate_trace":[{key:candidate[key] for key in trace_keys} for candidate in candidates],
                "candidate_max_cosine_error":max((h["cosine_error"] for h in candidates),default=0.),
                "reranker_model":RERANK_MODEL if self.retrieval=="rerank" else None,
                "reranker_revision":RERANK_REVISION if self.retrieval=="rerank" else None,
                "pair_max_tokens":pair_max_tokens,
                "cohort_join":"payload.profile_ids -> SQLite.cohort_profiles.profile_id",
                "interpretation":"Los perfiles describen cohortes portuguesas; la asociación con documentos colombianos es temática, sin enlace de personas ni inferencia causal."}

    @staticmethod
    def _relevance(hit,gold):
        if hit["doc_id"]!=gold["document_id"] or hit["page"] not in gold["pages"]:
            return False
        normalized=fold(hit["text"])
        if gold.get("evidence_regex"):
            return bool(re.search(fold(gold["evidence_regex"]),normalized,re.I|re.S))
        if gold.get("evidence_terms"):
            return any(fold(term) in normalized for term in gold["evidence_terms"])
        if gold.get("evidence_text"):
            return fold(gold["evidence_text"]) in normalized
        return True

    def evaluate(self,bench_filename="evaluation.json",prefix=""):
        """Evalúa un banco congelado y separa las salidas de cada partición/método."""
        if Path(bench_filename).name!=bench_filename or (prefix and not re.fullmatch(r"[A-Za-z0-9_]+",prefix)):
            raise ValueError("Use un nombre de banco y un prefijo de archivo simples.")
        benchmark_path=self.root/"data"/bench_filename
        benchmark=json.loads(benchmark_path.read_text())
        questions=benchmark["questions"]
        split=benchmark.get("split","development")
        if split=="holdout" and not prefix:
            raise ValueError("La reserva requiere prefix para conservar los resultados de desarrollo.")
        self.search("¿Cómo se organiza el seguimiento a la permanencia estudiantil?",3)
        records=[]
        search_results=[]
        all_latencies=[]
        for question in questions:
            repeats=[]
            responses=[]
            for _ in range(3):
                start=time.perf_counter()
                response=self.search(question["query"],3)
                repeats.append((time.perf_counter()-start)*1000)
                responses.append(response)
            all_latencies.extend(repeats)
            results=responses[0]
            first_signature=[h["chunk_id"] for h in results["hits"]]
            stable=all([h["chunk_id"] for h in r["hits"]]==first_signature for r in responses[1:])
            matched=set()
            relevance=[]
            for hit in results["hits"]:
                groups={g.get("evidence_group","answer") for g in question["gold"] if self._relevance(hit,g)}
                new=groups-matched
                relevance.append(int(bool(new)))
                matched.update(new)
            first=next((rank for rank,value in enumerate(relevance,1) if value),None)
            groups={g.get("evidence_group","answer") for g in question["gold"]}
            ideal=sum(1/np.log2(rank+1) for rank in range(1,min(3,len(groups))+1))
            dcg=sum(value/np.log2(rank+1) for rank,value in enumerate(relevance,1))
            record={"id":question["id"],"query":question["query"],"hit_at_3":int(first is not None),
                    "mrr_at_3":1/first if first else 0.,"ndcg_at_3":float(dcg/ideal if ideal else 0.),
                    "latency_p50_ms":float(np.median(repeats)),"latency_p95_ms":float(np.quantile(repeats,.95)),
                    "top1_score":results["hits"][0]["score"],"stable_top3":stable,
                    "ranking_method":self.retrieval,"source_scope_mode":results["source_scope"]["mode"],
                    "source_scope_document":results["source_scope"]["document_id"],
                    "relevant_ranks":";".join(str(i) for i,v in enumerate(relevance,1) if v)}
            if self.retrieval=="rerank":
                record["top1_rerank_score"]=results["hits"][0]["rerank_score"]
            records.append(record)
            search_results.append({"id":question["id"],"gold":question["gold"],"latencies_ms":repeats,**results})
        table=pd.DataFrame(records)
        metrics={"n_queries":len(questions),"latency_measurements":len(all_latencies),"repeats_per_query":3,
              "hit_at_3":float(table.hit_at_3.mean()),"mrr_at_3":float(table.mrr_at_3.mean()),
              "ndcg_at_3":float(table.ndcg_at_3.mean()),"latency_p50_ms":float(np.median(all_latencies)),
              "latency_p95_ms":float(np.quantile(all_latencies,.95)),"backend":self.backend,
              "stable_queries":int(table.stable_top3.sum()),"benchmark_sha256":file_hash(benchmark_path),
              "gold_unit":"documento_pagina_y_evidencia_del_pasaje_cuando_definida",
              "max_cosine_error":max(r["candidate_max_cosine_error"] for r in search_results),
              "ranking_method":self.retrieval,"candidate_pool":CANDIDATE_POOL if self.retrieval=="rerank" else 3,
              "reranker_model":RERANK_MODEL if self.retrieval=="rerank" else None,
              "reranker_revision":RERANK_REVISION if self.retrieval=="rerank" else None,
              "split":split,"benchmark_file":bench_filename,
              "scope":"banco_disenado_de_consultas; sin_generalizacion_estadistica"}
        stem=f"{prefix}_" if prefix else ""
        table.to_csv(self.output/f"{stem}evaluation.csv",index=False)
        (self.output/f"{stem}search_results.json").write_text(json.dumps(search_results,ensure_ascii=False,indent=2))
        versions={name:importlib.metadata.version(name) for name in ["numpy","pandas","PyMuPDF","torch","transformers","sentence-transformers","qdrant-client"]}
        retrieval_config={"ranking_method":self.retrieval,
                          "source_scope_policy":"explicit_unique_institution_alias",
                          "candidate_pool":CANDIDATE_POOL if self.retrieval=="rerank" else 3,
                          "reranker_title_context":self.retrieval=="rerank",
                          "pair_token_limit":PAIR_TOKEN_LIMIT if self.retrieval=="rerank" else None}
        comparison=None
        if self.retrieval=="rerank" and split!="holdout":
            for baseline_path in [self.output/"dense_comparison_summary.json",self.output/"baseline/summary.json"]:
                if baseline_path.exists():
                    baseline=json.loads(baseline_path.read_text())
                    old=baseline["metrics"]
                    same=(old.get("benchmark_sha256")==metrics["benchmark_sha256"]
                          and baseline.get("embedding_manifest")==self.embedding_manifest)
                    if same:
                        comparison={"baseline_file":str(baseline_path.relative_to(self.root)),
                            "baseline_method":"dense","current_method":"rerank","same_benchmark_and_embeddings":True,
                            "baseline_retrieval_config":baseline.get("retrieval_config",{"ranking_method":"dense","source_scope_policy":"all_sources"}),
                            "current_retrieval_config":retrieval_config,
                            "interpretation":"Comparacion de configuraciones completas; las diferencias de filtro y contexto deben considerarse al atribuir cambios.",
                            "metrics":{key:{"baseline":old[key],"current":metrics[key],"delta":metrics[key]-old[key]}
                                      for key in ["hit_at_3","mrr_at_3","ndcg_at_3","latency_p50_ms","latency_p95_ms"]}}
                        break
        summary={"dataset":self.dataset_summary,"cohort_profiles":self.cohort_profiles,"corpus":self.corpus_summary,
                 "chunks":self.chunk_stats,"index":self.index_report,"metrics":metrics,
                 "versions":versions,"embedding_manifest":self.embedding_manifest,"cache_reused":self.cache_reused,
                 "comparison_to_dense":comparison,"retrieval_config":retrieval_config}
        (self.output/f"{stem}summary.json").write_text(json.dumps(summary,ensure_ascii=False,indent=2))
        self.evaluation_runs[prefix or "development"]={"metrics":metrics,"records":records,
                                                       "search_results":search_results,"summary":summary}
        if not prefix:
            self.metrics=metrics
            self.evaluation_records=records
            self.search_results=search_results
            self.summary=summary
            self.negative_control=self.search("¿Cuál es la política de devolución de un teléfono comprado por Internet?",3)
            self.negative_control["scope_decision"]="fuera_del_dominio; no_se_genera_respuesta"
            (self.output/"out_of_scope.json").write_text(json.dumps(self.negative_control,ensure_ascii=False,indent=2))
        return metrics

    def run(self):
        self.ingest()
        self.prepare_corpus()
        self.encode()
        self.index()
        self.evaluate()
        return self.summary

    def close(self):
        if self.client is not None:
            self.client.close()
            self.client=None


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root",default=".")
    parser.add_argument("--backend",choices=["local","cloud"],default="local")
    parser.add_argument("--provider",choices=["e5","gemini2"],default="e5")
    parser.add_argument("--memory",action="store_true",help="Usar almacén local transitorio en vez de persistente.")
    parser.add_argument("--retrieval",choices=["dense","rerank"],default="rerank")
    parser.add_argument("--benchmark",default="evaluation.json")
    parser.add_argument("--prefix",default="")
    args=parser.parse_args()
    pipeline=StudentEvidencePipeline(args.root,args.backend,args.provider,persistent=not args.memory,retrieval=args.retrieval)
    try:
        pipeline.ingest()
        pipeline.prepare_corpus()
        pipeline.encode()
        pipeline.index()
        metrics=pipeline.evaluate(bench_filename=args.benchmark,prefix=args.prefix)
        print(json.dumps(metrics,ensure_ascii=False,indent=2))
    finally:
        pipeline.close()

if __name__=="__main__":
    main()
