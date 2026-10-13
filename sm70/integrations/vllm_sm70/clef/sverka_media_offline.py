# Запуск (CPU, venv боевого; Clef -- КОПИЯ каталога с joint_schema_model.py, не сам скачанный):
#   python -I sverka_media_offline.py <integrations/vllm_sm70> <копия clef> <модель>
# Ожидание: "ИТОГ: СОШЛОСЬ". Контроль 13.10: на версии без _токены_картинок падает TypeError MediaWithBytes.
# Офлайн-гейт (CPU) СКВОЗНОЙ цепочкой эндпоинта: _url -> загрузчик vLLM (MediaWithBytes) -> _токены_картинок
# -> _медиа_ид -> закодировать == encode_record Clef; развёртка vLLM тех же обёрток == записи эндпоинта.
import os, sys, io, base64, asyncio
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
sys.path.insert(0, sys.argv[1])            # integrations/vllm_sm70 кандидата
sys.path.insert(0, sys.argv[2])            # копия joint_schema_model.py Clef
MODEL = sys.argv[3]
from PIL import Image
import numpy as np
from transformers import AutoProcessor, AutoTokenizer, AutoImageProcessor
import fa2sm70_clef as F
import joint_schema_model as J
from vllm import envs
from vllm.multimodal.media import MEDIA_CONNECTOR_REGISTRY, MediaWithBytes

tok = AutoTokenizer.from_pretrained(MODEL)
proc = AutoProcessor.from_pretrained(MODEL)
ip = AutoImageProcessor.from_pretrained(MODEL)
con = MEDIA_CONNECTOR_REGISTRY.load(envs.VLLM_MEDIA_CONNECTOR, media_io_kwargs=None,
                                    allowed_local_media_path="", allowed_media_domains=None)
rng = np.random.default_rng(0)
def png(w, h):
    b = io.BytesIO(); Image.fromarray(rng.integers(0, 255, (h, w, 3), dtype=np.uint8)).save(b, "PNG")
    return base64.b64encode(b.getvalue()).decode()
records = [
    {"state": {"task": "Review the attached receipt.", "amount": 42}, "images": ["data:image/png;base64," + png(1280, 960)],
     "questions": {"legible": {"type": "noul", "instructions": "Is the receipt total legible?"},
                   "kind": {"type": "choice", "criteria": {"receipt": "A receipt", "photo": "A photo", "chart": "A chart"}}}},
    {"state": "two pictures", "images": [png(640, 480), {"image_url": {"url": "data:image/png;base64," + png(333, 777)}}],
     "questions": {"q": {"type": "score", "criteria": ["low", "mid", "high"]}}},
    {"state": "text only " * 50, "questions": {"u": {"type": "noul"}}},
]
ok = True
for i, r in enumerate(records):
    imgs = r.get("images") or []
    got = [asyncio.run(con.fetch_image_async(F._url(x))) for x in imgs]      # как эндпоинт
    obertka = all(isinstance(g, MediaWithBytes) for g in got)
    counts = F._токены_картинок(ip, got, {}) if got else []
    media = F._медиа_ид(tok, counts) if got else None
    ids, enc, eng = F.закодировать(tok, r, 16384, media)
    rc = dict(r, images=[g.media for g in got]) if got else r                    # Clef получает PIL тех же байтов
    ref = J.encode_record(tok, rc, max_length=16384, processor=proc if got else None)
    same_ids = list(ref.input_ids) == ids
    same_sp = [(q.question_span, list(q.option_spans), list(q.option_ids)) for q in ref.questions] == \
              [(qs, [tuple(x) for x in osp], list(oids)) for _q, _t, qs, osp, oids in enc]
    print(f"запись {i}: картинок {len(got)} (обёртка {obertka}) токенов {counts}, N={len(ids)}, движку {len(eng)}; "
          f"ids==Clef {same_ids}, спаны==Clef {same_sp}")
    ok &= same_ids and same_sp and (obertka or not got)
    r["_full"], r["_eng"], r["_got"] = ids, eng, got

from vllm.config import ModelConfig
from vllm.multimodal import MULTIMODAL_REGISTRY
mc = ModelConfig(model=MODEL, tokenizer=MODEL)
mp = MULTIMODAL_REGISTRY.create_processor(mc)
for i, r in enumerate(records):
    if not r["_got"]:
        continue
    items = mp.data_parser.parse_mm_data({"image": r["_got"]})                  # обёртки, как в multi_modal_data
    out = mp.apply(r["_eng"], items, hf_processor_mm_kwargs={})
    g = list(out["prompt_token_ids"])
    print(f"запись {i}: vLLM развернул {len(r['_eng'])} -> {len(g)}; == запись эндпоинта {g == r['_full']}; "
          f"сумма {F._сумма(g) == F._сумма(r['_full'])}")
    ok &= g == r["_full"]
print("ИТОГ:", "СОШЛОСЬ" if ok else "РАСХОЖДЕНИЕ")
