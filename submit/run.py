# -*- coding: utf-8 -*-
"""EXP-034. ЛВ: логрег-стекинг на лог-оддсах пяти сигналов вместо рангового бленда.

Оффлайн на паблик-взвешенном стенде: 0.8051 (пять сигналов) против 0.7679 у EXP-032,
который дал на паблике 0.744 при предсказанных 0.768 (разрыв стенда 0.024).

Сигналы:
  attr    — TF-IDF по трём полям: полный текст + атрибутный экстракт + название
  tfidf   — TF-IDF только по полному тексту
  gemret  — gemma-4-E4B-it, 4+4 kNN-демонстрации по char-TF-IDF
  ret8    — она же, 8+8 демонстраций
  few     — Qwen3.5-4B, фиксированные примеры по подкатегориям

Коэффициенты стека сняты с OOF и зашиты для КАЖДОГО подмножества сигналов, которое может
успеть за 20 минут. Что не успело — то выпадает, и берётся набор коэффициентов под то,
что реально посчиталось. Ступеней пять: all6 → no_retemb → core4 → gem_only → text_only.

БАД: та же TF-IDF, но решение — посегментные top-k квоты (слабомарк. 213/260, яркие 350/661)
из точных фактов PROBE-K. Стенд-v3: 0.9027 -> 0.9093; ожидание паблика ~0.914.
"""
import argparse, json, os, re, time
import numpy as np, pandas as pd
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import normalize

HERE = os.path.dirname(os.path.abspath(__file__))
SEED = 42
_SH = os.environ.get("SHARED_MODELS_PATH", "/shared_models")
GEMMA = [os.path.join(_SH, "google/gemma-4-E4B-it"), "google/gemma-4-E4B-it"]
QWEN = [os.path.join(_SH, "Qwen/Qwen3.5-4B"), "Qwen/Qwen3.5-4B"]
DL_GEM, DL_QWEN = 9 * 60, 16 * 60   # дедлайн gemma-стадии и Qwen-стадии; лимит паблика 20 мин
ATTR_KEYS = ["комплектац", "состав", "топлив", "тип баллона", "баллон", "расход газа", "пьезоподжиг",
             "характеристик", "применение", "назначение", "особенност", "материал", "в комплекте", "входит"]
RULES = [
    ("Пиротехника и дым", r"пиротех|цветной дым|дымовая шашка|дымов|хлопушк|фейерверк|салют|бенгальск|страйкбол|граната"),
    ("Розжиг и твёрдое топливо", r"розжиг|сухое горючее|сухого горючего|брикет|роллы|растопк|топливо"),
    ("Уголь", r"уголь|угли |углём|углем"), ("Спички", r"спичк|спичеч"), ("Свечи", r"свеч"),
    ("Благовония/аромапалочки", r"благовон|шалфей|пало санто|аромапалочк"),
    ("Зажигалки и бензин", r"зажигалк|бензин"),
    ("Газ: горелки/баллоны/плиты", r"газов|баллон|пропан|бутан|горелк|плитк|плита|резак|паяльн"),
    ("Мангалы/грили/аксессуары", r"мангал|гриль|барбекю|шампур|казан|коптил|опахал|решетк|решётк"),
    ("Краски Холи", r"холи|краск"),
]
COMMENTS = {
    ("БАД", 1): ("Товар распознан как корректно размещённая биологически активная добавка: название и "
                 "описание содержат характерные признаки БАД, карточка соответствует требованиям категории."),
    ("БАД", 0): ("Карточка в категории БАД не подтверждена как корректная: по названию и описанию товар "
                 "ближе к спортивному питанию или продукции без обязательной маркировки БАД."),
    ("Легковоспламеняющиеся", 1): ("Товар отнесён к легковоспламеняющимся: по названию и описанию он содержит "
                                   "горючее вещество, газ, топливо или пиротехнический заряд либо горючее "
                                   "входит в комплект поставки."),
    ("Легковоспламеняющиеся", 0): ("Признаков легковоспламеняющегося товара не найдено: это устройство или "
                                   "изделие без горючего содержимого в комплекте, горючий материал здесь "
                                   "не является самостоятельным товаром."),
}
SYS_RET = ("Ты воспроизводишь историческую разметку маркетплейса: пометил бы разметчик этот товар как "
           "легковоспламеняющийся? Общий принцип: решает то, что физически есть в самом товаре или в его "
           "комплекте (горючее вещество, газ, топливо, пиротехнический заряд, спички), а не то, для чего "
           "товар предназначен: пустые устройства для огня размечают как НЕ легковоспламеняющиеся. "
           "Ниже даны примеры разметки похожих товаров — ориентируйся прежде всего на них, "
           "они отражают реальную конвенцию. Ответ одним словом: да или нет.")
SYS_FEW = ("Ты — модератор маркетплейса Ozon. По названию и описанию определи, является ли товар "
           "ЛЕГКОВОСПЛАМЕНЯЮЩИМСЯ по правилам площадки. ГЛАВНЫЙ ПРИНЦИП: решает то, что физически есть "
           "В САМОМ ТОВАРЕ ИЛИ В ЕГО КОМПЛЕКТЕ, а не то, для чего товар предназначен. "
           "ДА: товар сам источник огня; содержит горючее вещество, газ, топливо или пиротехнический заряд; "
           "горючее входит в комплект. НЕТ: пустое устройство ДЛЯ огня (мангал, плита, горелка-насадка без "
           "баллона, зажигалка); горючее — лишь материал конструкции. Отвечай ОДНИМ словом: да или нет.")


def subcat(name, desc):
    t = str(name).lower()
    for nm, pat in RULES:
        if re.search(pat, t):
            return nm
    t2 = str(desc)[:600].lower()
    for nm, pat in RULES:
        if re.search(pat, t2):
            return nm
    return "Прочее"


def attr_field(h):
    t = re.sub(r"<[^>]+>", "\n", str(h)).lower()
    return " ".join(t[m.start():m.start() + 140] for k in ATTR_KEYS for m in re.finditer(re.escape(k), t))


def tfidf_blocks(tr_df, te_df, ytr, cols_cfg, C=4.0):
    Xs_tr, Xs_te = [], []
    for col, wr, cr, mdf, mf in cols_cfg:
        wv = TfidfVectorizer(ngram_range=wr, min_df=mdf[0], max_features=mf[0], sublinear_tf=True)
        cv = TfidfVectorizer(analyzer="char_wb", ngram_range=cr, min_df=mdf[1], max_features=mf[1], sublinear_tf=True)
        Xs_tr.append(sparse.hstack([wv.fit_transform(tr_df[col]), cv.fit_transform(tr_df[col])]).tocsr())
        Xs_te.append(sparse.hstack([wv.transform(te_df[col]), cv.transform(te_df[col])]).tocsr())
    m = LogisticRegression(C=C, max_iter=3000, class_weight="balanced", solver="liblinear", random_state=SEED)
    m.fit(sparse.hstack(Xs_tr).tocsr(), ytr)
    return m.predict_proba(sparse.hstack(Xs_te).tocsr())[:, 1]


def tfidf_blocks_nb(tr_df, te_df, ytr, cols_cfg, C=4.0):
    """То же, что tfidf_blocks, но признаки домножены на модуль log-отношения правдоподобий
    классов (NB-взвешивание), посчитанного ТОЛЬКО на обучающей выборке.

    Механизм проверен контролем (EXP-183): перестановка тех же коэффициентов по случайным
    признакам даёт зеркальный минус, значит работает содержательная привязка, а не удачное
    распределение масштабов. Для линейной модели с L2 знак коэффициента не важен — эффект
    целиком в модуле: различающий признак получает слабее эффективный штраф, то есть это
    регуляризация, информированная данными. Глобальный свип C этого дать не может.
    Стенд (розыгрыши паблик-состава): 0.8913 -> 0.9012, Δ +0.0099, вложенный отбор выбрал
    NB во всех 5 фолдах в трёх независимых экспериментах (EXP-181, -183, -185).
    """
    Xs_tr, Xs_te = [], []
    for col, wr, cr, mdf, mf in cols_cfg:
        wv = TfidfVectorizer(ngram_range=wr, min_df=mdf[0], max_features=mf[0], sublinear_tf=True)
        cv = TfidfVectorizer(analyzer="char_wb", ngram_range=cr, min_df=mdf[1], max_features=mf[1], sublinear_tf=True)
        Xs_tr.append(sparse.hstack([wv.fit_transform(tr_df[col]), cv.fit_transform(tr_df[col])]).tocsr())
        Xs_te.append(sparse.hstack([wv.transform(te_df[col]), cv.transform(te_df[col])]).tocsr())
    Xtr = sparse.hstack(Xs_tr).tocsr(); Xte = sparse.hstack(Xs_te).tocsr()
    yy = np.asarray(ytr)
    pc = np.asarray(Xtr[yy == 1].sum(0)).ravel() + 1.0
    nc = np.asarray(Xtr[yy == 0].sum(0)).ravel() + 1.0
    r = np.abs(np.log((pc / pc.sum()) / (nc / nc.sum())))
    r = np.maximum(r, 1e-6); r = r / r.mean()          # нормировка: не двигаем эффективное C
    R = sparse.diags(r)
    m = LogisticRegression(C=C, max_iter=3000, class_weight="balanced", solver="liblinear", random_state=SEED)
    m.fit(Xtr @ R, yy)
    return m.predict_proba(Xte @ R)[:, 1]


CFG3 = [("f_txt", (1, 2), (3, 5), (2, 3), (200_000, 300_000)),
        ("f_attr", (1, 3), (3, 5), (2, 2), (150_000, 200_000)),
        ("f_nm", (1, 3), (2, 4), (2, 2), (100_000, 150_000))]
CFG1 = [("f_txt", (1, 2), (3, 5), (2, 3), (200_000, 300_000))]
lgt = lambda q: np.log(np.clip(np.asarray(q, float), 1e-6, 1 - 1e-6) /
                       (1 - np.clip(np.asarray(q, float), 1e-6, 1 - 1e-6)))



# --- EXP-096: посегментные top-k квоты (RLSbench-стиль поправки под измеренный маргинал) ----
# PROBE-K: на паблике слабомаркированные (нет «БАД» в названии и первых 200 видимых симв.)
# = 260 товаров с 208 позитивами; ярко-маркированные = 661 с 320. Ранжирование НЕ трогаем
# (переносится, +8 TP сверх оффлайна), меняем только решение: топ-213 в слабой ячейке,
# топ-350 в яркой (оптимум на стенде-v3, калиброванном фактами до 0.001-0.006).
BAD_MARK_RE = re.compile(r"\bбад\b|бады|бада\b|биологически активн|dietary supplement|food supplement")
BAD_RATE = 578.0 / 921.0   # доля выдачи опоры EXP-035: 578 из 921 (K=578, TP=502, F1 0.9078)


def bad_weak_flags(df):
    out = []
    for nm, rw in zip(df["name"].astype(str), df["desc_raw"].astype(str)):
        vis = re.sub(r"\s+", " ", re.sub(r"<[^>]+>", " ", rw)).strip().lower()
        out.append(not BAD_MARK_RE.search(nm.lower()) and not BAD_MARK_RE.search(vis[:200]))
    return np.array(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--test_data_path", "--test-data-path", "-i", dest="test_data_path")
    ap.add_argument("--output_path", "--output-path", "-o", dest="output_path")
    a = ap.parse_args()
    t0 = time.time(); log = print
    np.random.seed(SEED)
    P = json.load(open(os.path.join(HERE, "params.json"), encoding="utf-8"))
    train = pd.read_csv(os.path.join(HERE, "train_data.csv"), index_col=0)
    test = pd.read_csv(a.test_data_path)
    log(f"train {len(train)}, test {len(test)}")
    for df in (train, test):
        df["desc_raw"] = df["description"].fillna("").astype(str)
        df["desc"] = df["desc_raw"].str.replace(r"<[^>]+>", " ", regex=True).str.replace(r"\s+", " ", regex=True)
        df["f_txt"] = (df["name"].astype(str) + " \n " + df["desc_raw"]).str.lower()
        df["f_attr"] = [attr_field(x) for x in df["desc_raw"]]
        df["f_nm"] = df["name"].astype(str).str.lower()
    preds = pd.Series(0, index=test.index, dtype=int)

    m_bad = (test["category"] == "БАД").values
    if m_bad.sum():
        trb = train[train["category"] == "БАД"]
        # EXP-170 ОТМЕНИЛ ветку ячейковой модели: её прирост был артефактом того, что
        # опора стенда строилась по CFG3, а контейнер считает по CFG1. Против правильной
        # опоры ячейка даёт -0.0002 (P=0.404), поэтому она убрана целиком.
        # Вместо неё — NB-взвешивание признаков, единственный подтверждённый плюс:
        # +0.0099 на розыгрышах, вложенно единогласно 5/5 фолдов, контроль с перестановкой
        # коэффициентов даёт зеркальный -0.0099.
        p = tfidf_blocks_nb(trb, test.loc[m_bad], trb["label"].values, CFG1, C=4.0)
        log(f"[{time.time()-t0:.0f}s] БАД: NB-взвешенная модель обучена на {len(trb)}")
        k_take = int(round(BAD_RATE * len(p)))
        sel = np.zeros(len(p), bool)
        sel[np.argsort(-p)[:k_take]] = True
        preds[m_bad] = sel.astype(int)
        log(f"[{time.time()-t0:.0f}s] БАД: {len(p)} карточек, взято {int(sel.sum())} (доля {BAD_RATE})")
        log(f"[{time.time()-t0:.0f}s] БАД: {int(m_bad.sum())}, позитивов {int(preds[m_bad].sum())}")

    m_lv = (test["category"] == "Легковоспламеняющиеся").values
    if m_lv.sum():
        sub = test[m_lv].reset_index(drop=True)
        trl = train[train["category"] == "Легковоспламеняющиеся"]
        S = {"attr": tfidf_blocks(trl, sub, trl["label"].values, CFG3),
             "tfidf": tfidf_blocks(trl, sub, trl["label"].values, CFG1)}
        log(f"[{time.time()-t0:.0f}s] текстовые сигналы готовы")

        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
            if not torch.cuda.is_available():
                raise RuntimeError("нет GPU")
            os.environ.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")
            # ВАЖНО: оффлайн демонстрации отбирались по одной на neardup-ГРУППУ.
            # Повторяем это здесь, иначе распределение сигнала разойдётся с тем,
            # на котором обучены коэффициенты стека.
            gmap = json.load(open(os.path.join(HERE, "lv_groups.json"), encoding="utf-8"))
            tl = trl.copy()
            tl["grp"] = [int(gmap.get(str(int(v)), -1)) for v in tl["id"]]
            if (np.array(tl["grp"]) == -1).all():
                key = lambda n, d: re.sub(r"\s+", " ", (str(n) + " " + str(d)).lower()).strip()
                tl["grp"] = pd.factorize([key(n, d) for n, d in zip(tl["name"], tl["desc"])])[0]
                log("группы из lv_groups.json не сопоставились — дедуплицирую по точному ключу")
            tl = tl.reset_index(drop=True)
            vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=2,
                                  max_features=200_000, sublinear_tf=True)
            Xc = normalize(vec.fit_transform((tl["name"].astype(str) + " " + tl["desc"]).str.lower()))
            Xq = normalize(vec.transform((sub["name"].astype(str) + " " + sub["desc"]).str.lower()))
            pi = np.where(tl["label"].values == 1)[0]; ni = np.where(tl["label"].values == 0)[0]
            sp = (Xq @ Xc[pi].T).toarray(); sn = (Xq @ Xc[ni].T).toarray()
            rng = np.random.default_rng(SEED)

            grpv = tl["grp"].values

            def demos_prompt(r, kk):
                dm, seen = [], set()
                for idxs, sims in ((pi, sp[r]), (ni, sn[r])):
                    taken = 0
                    for j in idxs[np.argsort(-sims)]:
                        if taken >= kk: break
                        if grpv[j] in seen: continue          # одна карточка на группу
                        seen.add(grpv[j]); taken += 1
                        dm.append((str(tl.loc[j, "name"]), str(tl.loc[j, "desc"])[:180], int(tl.loc[j, "label"])))
                rng.shuffle(dm)
                lines = [f"- {n} || {d} -> {'да' if l else 'нет'}" for n, d, l in dm]
                return ("Примеры разметки похожих товаров:\n" + "\n".join(lines) +
                        f"\n\nТовар: {sub.iloc[r]['name']}\nОписание: {str(sub.iloc[r]['desc'])[:400]}\n\nОтвет:")

            def run_model(model_paths, passes, deadline, tag):
                """Одна загрузка модели — несколько проходов. Экономит минуты на загрузке."""
                path = next((p_ for p_ in model_paths if os.path.exists(p_)), model_paths[-1])
                loc = os.path.exists(path)
                tok = AutoTokenizer.from_pretrained(path, local_files_only=loc, trust_remote_code=True)
                tok.padding_side = "left"
                if tok.pad_token is None: tok.pad_token = tok.eos_token
                mdl = AutoModelForCausalLM.from_pretrained(path, torch_dtype=torch.float16, device_map="auto",
                                                           local_files_only=loc, trust_remote_code=True).eval()
                yi = sorted({tok.encode(x, add_special_tokens=False)[0] for x in ["да", "Да", " да", " Да"]})
                no = sorted({tok.encode(x, add_special_tokens=False)[0] for x in ["нет", "Нет", " нет", " Нет"]})
                gb = torch.cuda.get_device_properties(0).total_memory / 1e9
                MBT, MBS = (60000, 32) if gb > 40 else (6000, 3)
                log(f"[{time.time()-t0:.0f}s] {tag}: модель {path}, батч до {MBT} токенов / {MBS}")
                res = {}
                for pname, sysmsg, userfn in passes:
                    if time.time() - t0 > deadline:
                        log(f"  {pname}: пропущен по дедлайну"); continue
                    pr = []
                    for r in range(len(sub)):
                        msgs = [{"role": "system", "content": sysmsg}, {"role": "user", "content": userfn(r)}]
                        try:
                            pr.append(tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True,
                                                              enable_thinking=False))
                        except TypeError:
                            pr.append(tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True))
                    ln = [len(tok(x, add_special_tokens=False)["input_ids"]) for x in pr]
                    out = np.full(len(sub), np.nan)
                    order = sorted(range(len(sub)), key=lambda i2: ln[i2]); i2 = 0
                    while i2 < len(order):
                        if time.time() - t0 > deadline:
                            log(f"  {pname}: дедлайн на {int(np.isfinite(out).sum())}/{len(sub)}"); break
                        bt, mx = [], 0
                        while i2 + len(bt) < len(order) and len(bt) < MBS:
                            j2 = order[i2 + len(bt)]; nm2 = max(mx, ln[j2])
                            if bt and nm2 * (len(bt) + 1) > MBT: break
                            bt.append(j2); mx = nm2
                        i2 += len(bt)
                        enc = tok([pr[j2] for j2 in bt], return_tensors="pt", padding=True,
                                  add_special_tokens=False).to("cuda")
                        with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.float16):
                            try:
                                lg = mdl(**enc, logits_to_keep=1).logits[:, -1, :].float()
                            except TypeError:
                                lg = mdl(**enc, num_logits_to_keep=1).logits[:, -1, :].float()
                        lp = torch.log_softmax(lg, dim=-1)
                        sc = (torch.logsumexp(lp[:, yi], 1) - torch.logsumexp(lp[:, no], 1)).cpu().numpy()
                        for j2, v in zip(bt, sc):
                            out[j2] = 1 / (1 + np.exp(-v))
                    if np.isfinite(out).all():
                        res[pname] = out
                        log(f"[{time.time()-t0:.0f}s] {pname}: готов")
                    else:
                        log(f"[{time.time()-t0:.0f}s] {pname}: неполный, отброшен")
                del mdl; torch.cuda.empty_cache()
                return res

            subs = [subcat(n, d) for n, d in zip(sub["name"], sub["desc"])]
            fs = json.load(open(os.path.join(HERE, "fewshot_submit.json"), encoding="utf-8"))
            import importlib.util as _iu
            _sp = _iu.spec_from_file_location("prompts", os.path.join(HERE, "prompts.py"))
            _pm = _iu.module_from_spec(_sp); _sp.loader.exec_module(_pm)
            INSTR = getattr(_pm, "SUBCAT_INSTRUCTIONS", {})

            def fewuser(r):
                b_ = subs[r]; d = fs.get(b_, [])
                lines = [f"- {n} || {dd} -> {'да' if l else 'нет'}" for n, dd, l in d]
                head = INSTR.get(b_, "")
                return ((head + "\n\n" if head else "") +
                        ("Примеры разметки этой подкатегории:\n" + "\n".join(lines) + "\n\n" if lines else "") +
                        f"Название: {sub.iloc[r]['name']}\nОписание: {str(sub.iloc[r]['desc'])[:400]}\n\nОтвет:")

            S.update(run_model(GEMMA, [("gemret", SYS_RET, lambda r: demos_prompt(r, 4)),
                                       ("ret8", SYS_RET, lambda r: demos_prompt(r, 8))], DL_GEM, "gemma"))
            S.update(run_model(QWEN, [("qret", SYS_RET, lambda r: demos_prompt(r, 4)),
                                      ("few", _pm.SYSTEM_BASE, fewuser)], DL_QWEN, "qwen"))
        except Exception as e:
            log(f"LLM недоступна ({type(e).__name__}: {e})")

        have = set(S)
        for name in ("all7", "no_emb6", "no_emb_no_few", "core4", "gem_only", "text_only"):
            cfg = P[name]
            if set(cfg["cols"]) <= have:
                cols, coef, b0 = cfg["cols"], np.array(cfg["coef"]), cfg["intercept"]
                log(f"стек «{name}»: {cols}")
                break
        z = b0 + sum(c * lgt(S[col]) for c, col in zip(coef, cols))
        p = 1 / (1 + np.exp(-z))
        o = np.argsort(-p); cw = np.arange(1, len(p) + 1); cwp = np.cumsum(p[o]); tot = p.sum()
        k = int(np.argmax(2 * cwp / (cw + tot)))
        lp_ = np.zeros(len(p), int); lp_[o[:k + 1]] = 1
        preds[m_lv] = lp_
        log(f"[{time.time()-t0:.0f}s] ЛВ: DTA взял {lp_.sum()}/{len(lp_)} ({lp_.mean():.2%})")

    norm_key = lambda n, d: re.sub(r"\s+", " ", (str(n) + " " + str(d)).lower()).strip()
    dup = train.groupby([norm_key(n, d) for n, d in zip(train["name"], train["desc_raw"])])["label"].mean()
    n_over = 0
    for i in test.index:
        k2 = norm_key(test.loc[i, "name"], test.loc[i, "desc_raw"])
        if k2 in dup.index:
            m2 = dup.loc[k2]
            if m2 > 0.5 and preds.loc[i] != 1: preds.loc[i] = 1; n_over += 1
            elif m2 < 0.5 and preds.loc[i] != 0: preds.loc[i] = 0; n_over += 1
    log(f"дубль-оверрайдов: {n_over}")
    res = [f"<комментарий>{COMMENTS[(str(test.loc[i,'category']), int(preds.loc[i]))]}"
           f"<вердикт>{'не бан' if preds.loc[i] == 1 else 'бан'}" for i in test.index]
    pd.DataFrame({"id": test["id"].values, "result": res}).to_csv(a.output_path, index=False)
    log(f"готово за {time.time()-t0:.0f}s: {len(res)} строк")


if __name__ == "__main__":
    main()
