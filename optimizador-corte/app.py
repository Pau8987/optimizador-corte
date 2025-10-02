import os, re, math
from collections import Counter, defaultdict
from typing import List, Tuple, Dict, Optional

import pandas as pd
import gradio as gr

# =========================================================
# Utils
# =========================================================

def parse_piezas_con_parentesis(texto: str) -> List[float]:
    """
    '300(2), 400; 500(3) 450 240(4)' -> lista plana.
    Acepta coma o punto como separador decimal: 500,5 o 500.5
    """
    if not str(texto).strip():
        return []
    toks = re.split(r"[,\s;]+", str(texto).strip())
    out: List[float] = []
    pat = re.compile(r"^(\d+(?:[.,]\d+)?)(?:\((\d+)\))?$")
    for tk in toks:
        if not tk:
            continue
        m = pat.match(tk)
        if not m:
            raise ValueError(f"Entrada inválida: '{tk}'. Usa 300 o 300(2).")
        L = float(m.group(1).replace(",", "."))
        q = int(m.group(2)) if m.group(2) else 1
        if L <= 0 or q <= 0:
            raise ValueError(f"Valores no válidos en '{tk}'.")
        out += [L] * q
    return out

def _fmt_num(x: float) -> str:
    return str(int(x)) if abs(x - round(x)) < 1e-9 else f"{x:.3f}"

def cortes_compactos(cuts: List[float]) -> str:
    """500(2) 400 300(3) — agrupa con cantidad y ordena desc."""
    cnt = Counter(round(c, 6) for c in cuts)
    return " ".join(
        f"{_fmt_num(L)}({q})" if q > 1 else _fmt_num(L)
        for L, q in sorted(cnt.items(), key=lambda t: -t[0])
    )

# Orden global solicitado:
# 1) barras que contienen la longitud mayor global primero;
# 2) dentro de ellas, por nº de cortes de esa longitud (desc);
# 3) repetir con la 2ª mayor, etc.
def barra_sort_key_global(cuts: List[float], longitudes_globales_desc: List[float]) -> Tuple:
    cnt = Counter(round(c, 6) for c in cuts)
    key: List[float] = []
    for L in longitudes_globales_desc:
        key.append(-1.0 if cnt[L] > 0 else 0.0)  # barras con L primero
        key.append(-float(cnt[L]))               # más cantidad de L, antes
    # desempates:
    key.append(-sum(cnt.values()))               # más piezas totales, antes
    key.append(-max(cnt.keys()) if cnt else 0.0) # mayor pieza en la barra
    return tuple(key)

# =========================================================
# Excel (solo hoja Piezas)
# =========================================================

def _coerce_len(x) -> float:
    if pd.isna(x): raise ValueError("Hay longitudes vacías en el Excel.")
    s = str(x).strip()
    if not s: raise ValueError("Hay longitudes vacías en el Excel.")
    return float(s.replace(",", "."))

def leer_excel_piezas(path: str) -> pd.DataFrame:
    """
    Hoja obligatoria: 'Piezas'
    Columnas (ignora mayúsculas/espacios): longitud_mm, cantidad
    """
    xls = pd.ExcelFile(path)
    sheets = {s.lower(): s for s in xls.sheet_names}
    if "piezas" not in sheets:
        raise ValueError("El Excel debe tener una hoja llamada 'Piezas'.")
    df = pd.read_excel(path, sheet_name=sheets["piezas"])

    cols_norm = {c: re.sub(r"[^a-z]", "", c.lower()) for c in df.columns.astype(str)}
    inv = {v: k for k, v in cols_norm.items()}
    col_len = inv.get("longitudmm") or inv.get("longitud") or inv.get("largo") or inv.get("l")
    col_qty = inv.get("cantidad") or inv.get("qty") or inv.get("q")
    if not col_len or not col_qty:
        raise ValueError("Faltan columnas: 'longitud_mm' y 'cantidad' en la hoja Piezas.")

    df = df[[col_len, col_qty]].copy()
    df.columns = ["longitud_mm", "cantidad"]
    df["longitud_mm"] = df["longitud_mm"].map(_coerce_len)
    df["cantidad"] = df["cantidad"].astype(int)
    if (df["longitud_mm"] <= 0).any() or (df["cantidad"] <= 0).any():
        raise ValueError("Las longitudes y cantidades deben ser > 0.")
    return df

def piezas_desde_df(df_pzs: pd.DataFrame) -> List[float]:
    piezas: List[float] = []
    for _, r in df_pzs.iterrows():
        piezas += [float(r["longitud_mm"])] * int(r["cantidad"])
    return piezas

# =========================================================
# Agrupar tipos y voraz (para cota superior)
# =========================================================

def agrupar_tipos(piezas: List[float], dec: int = 6) -> Tuple[List[float], List[int]]:
    cnt: Dict[float, int] = defaultdict(int)
    for p in piezas:
        cnt[round(float(p), dec)] += 1
    tipos_L = sorted(cnt.keys(), reverse=True)   # grande->pequeño
    tipos_Q = [cnt[L] for L in tipos_L]
    return tipos_L, tipos_Q

def greedy_bins_agg(L: float, tipos_L: List[float], tipos_Q: List[int], kerf: float) -> List[List[int]]:
    """
    Voraz para UPPER BOUND: acomoda tipos en barras maximizando uso.
    Opera en capacidad efectiva: (L+kerf) y coste (L_t+kerf).
    """
    T = len(tipos_L)
    rem = tipos_Q[:]
    bars: List[List[int]] = []
    caps: List[float] = []

    order = list(range(T))
    order.sort(key=lambda t: tipos_L[t], reverse=True)

    while sum(rem) > 0:
        bars.append([0]*T)
        caps.append(L + kerf)
        j = len(bars) - 1
        for t in order:
            w = tipos_L[t] + kerf
            if w <= 0: continue
            max_fit = int(math.floor(caps[j] / w))
            if max_fit <= 0 or rem[t] <= 0: continue
            use = min(rem[t], max_fit)
            bars[j][t] += use
            rem[t] -= use
            caps[j] -= use * w

    return [row for row in bars if sum(row) > 0]

# =========================================================
# MILP: SOLO minimizar nº de barras (factibilidad con B fijo)
# =========================================================

def solve_fixed_B(
    L: float, tipos_L: List[float], tipos_Q: List[int], kerf: float,
    B: int, time_limit: int = 20, threads: int = 4
) -> Optional[List[List[int]]]:
    """
    Devuelve asignación por barra/tipo si es factible con B barras, si no devuelve None.
    """
    try:
        import pulp
    except Exception:
        raise RuntimeError("PuLP no disponible. Añádelo a requirements.txt (pulp>=2.7).")

    T = len(tipos_L)
    J = range(B)
    Tset = range(T)

    prob = pulp.LpProblem("CS_FixedB_Agg", pulp.LpStatusOptimal)
    x = {(t, j): pulp.LpVariable(f"x_{t}_{j}", lowBound=0, upBound=tipos_Q[t], cat=pulp.LpInteger)
         for t in Tset for j in J}
    s = {j: pulp.LpVariable(f"s_{j}", lowBound=0) for j in J}

    # Demanda por tipo
    for t in Tset:
        prob += pulp.lpSum(x[(t, j)] for j in J) == tipos_Q[t]

    # Capacidad y límites por tipo en cada barra
    for j in J:
        prob += s[j] == pulp.lpSum((tipos_L[t] + kerf) * x[(t, j)] for t in Tset)
        prob += s[j] <= (L + kerf)
        for t in Tset:
            max_units = int(math.floor((L + kerf) / (tipos_L[t] + kerf)))
            prob += x[(t, j)] <= max_units

    # Orden de uso no creciente (rompe simetría)
    for j in range(B - 1):
        prob += s[j] >= s[j + 1]

    prob += 0  # solo factibilidad

    solver = pulp.PULP_CBC_CMD(msg=False, timeLimit=time_limit, threads=threads)
    prob.solve(solver)
    if pulp.LpStatus[prob.status] not in ("Optimal", "Feasible"):
        return None

    cuts_idx: List[List[int]] = [[0]*T for _ in J]
    for j in J:
        for t in Tset:
            v = x[(t, j)].value()
            cuts_idx[j][t] = int(round(v)) if v is not None else 0
    return [row for row in cuts_idx if sum(row) > 0]

def resolver_min_barras(L: float, piezas: List[float], kerf: float) -> Tuple[List[List[float]], bool]:
    """
    Devuelve (plan, es_optimo).
      - LB fuerte
      - UB por voraz
      - prueba B=LB..UB con CBC rápido; devuelve la primera factible (óptima).
      - si ninguna factible en el tiempo, devuelve el voraz (no óptimo).
    """
    # piezas exactamente iguales a L: cada una ocupa una barra sola (no mezclan)
    piezas_rest = [p for p in piezas if abs(p - L) > 1e-9]
    plan_pre = [[L] for p in piezas if abs(p - L) <= 1e-9]

    tipos_L, tipos_Q = agrupar_tipos(piezas_rest)
    if sum(tipos_Q) == 0:
        return plan_pre, True

    T = len(tipos_L)
    Qtot = sum(tipos_Q)
    sum_len = sum(tipos_L[t]*tipos_Q[t] for t in range(T))
    LB = math.ceil((sum_len + kerf * Qtot) / (L + kerf))

    greedy_assign = greedy_bins_agg(L, tipos_L, tipos_Q, kerf)
    UB = len(greedy_assign)

    best_assign = None
    for B in range(LB, UB + 1):
        assign = solve_fixed_B(L, tipos_L, tipos_Q, kerf, B, time_limit=20, threads=4)
        if assign is not None:
            best_assign = assign
            UB = B
            break

    if best_assign is None:
        best_assign = greedy_assign
        es_optimo = False
    else:
        es_optimo = True

    # reconstruye plan
    plan: List[List[float]] = []
    for row in best_assign:
        barra = []
        for t, q in enumerate(row):
            if q > 0:
                barra += [tipos_L[t]] * q
        if barra:
            barra.sort(reverse=True)
            plan.append(barra)

    plan = plan_pre + plan
    return plan, es_optimo

# =========================================================
# Compactador heurístico (quita barras si es posible)
# =========================================================

def cabe_en_barra(cortes: List[float], L: float, kerf: float) -> bool:
    if not cortes: return True
    usado = sum(cortes) + (len(cortes) - 1) * kerf
    return usado <= L + 1e-9

def intenta_colocar_en_barras(resto: List[float], barras: List[List[float]], L: float, kerf: float) -> bool:
    piezas = sorted(resto, reverse=True)
    for pieza in piezas:
        mejor_j, mejor_hueco = None, None
        for j, cuts in enumerate(barras):
            candidato = cuts + [pieza]
            if cabe_en_barra(candidato, L, kerf):
                usado = sum(candidato) + (len(candidato) - 1) * kerf
                hueco = L - usado
                if (mejor_hueco is None) or (hueco < mejor_hueco - 1e-9):
                    mejor_j, mejor_hueco = j, hueco
        if mejor_j is None:
            return False
        barras[mejor_j].append(pieza)
        barras[mejor_j].sort(reverse=True)
    return True

def compactar_plan(plan: List[List[float]], L: float, kerf: float) -> List[List[float]]:
    plan = [sorted(b, reverse=True) for b in plan]
    cambio = True
    while cambio and len(plan) > 1:
        cambio = False
        # ordenar por uso ascendente para intentar quitar la más floja primero
        plan.sort(key=lambda cuts: (sum(cuts) + (len(cuts) - 1) * kerf))
        ultima = plan[-1][:]
        resto = plan[:-1]
        if intenta_colocar_en_barras(ultima, resto, L, kerf):
            plan = resto
            cambio = True
    return plan

# =========================================================
# Agregados / Orden / Orquestación
# =========================================================

def ordenar_barras(plan: List[List[float]]) -> List[List[float]]:
    longitudes_globales = sorted({round(c, 6) for cuts in plan for c in cuts}, reverse=True)
    return sorted(plan, key=lambda cuts: barra_sort_key_global(cuts, longitudes_globales))

def resumen_por_longitud(plan: List[List[float]]) -> pd.DataFrame:
    cnt = Counter(round(c, 6) for cuts in plan for c in cuts)
    filas = [{"Longitud (mm)": _fmt_num(L), "Cantidad total": q}
             for L, q in sorted(cnt.items(), key=lambda t: -t[0])]
    return pd.DataFrame(filas)

def barras_por_patron(plan: List[List[float]]) -> pd.DataFrame:
    patrones: Dict[str, List[int]] = defaultdict(list)
    for i, cuts in enumerate(plan, start=1):
        patrones[cortes_compactos(cuts)].append(i)
    filas = [{"Patrón cortes": k, "Nº barras": len(v), "Barras #": ", ".join(map(str, v))}
             for k, v in sorted(patrones.items(), key=lambda t: (-len(t[1]), t[0]))]
    return pd.DataFrame(filas)

def optimizar(L: float, piezas: List[float], kerf: float):
    if L <= 0: raise ValueError("La longitud de barra debe ser > 0.")
    if kerf < 0: raise ValueError("El kerf debe ser ≥ 0.")
    if any(p <= 0 for p in piezas): raise ValueError("Todas las piezas deben ser > 0.")
    if any(p > L + 1e-9 for p in piezas):
        raise ValueError(f"Hay pieza {max(piezas)} mm mayor que la barra {L} mm.")

    plan, es_optimo = resolver_min_barras(L, piezas, kerf)

    # Compactar y ordenar
    plan = compactar_plan(plan, L, kerf)
    plan = ordenar_barras(plan)

    filas = []
    total_usado = 0.0
    for i, cuts in enumerate(plan, start=1):
        usado = sum(cuts) + (len(cuts) - 1) * kerf if cuts else 0.0
        total_usado += usado
        uso_pct = max(0.0, min(100.0, 100.0 * usado / L))
        filas.append({
            "Barra #": i,
            "Cortes (mm)": cortes_compactos(cuts),
            "Uso (mm)": round(usado, 3),
            "Uso (%)": round(uso_pct, 2),
            "Desperdicio (mm)": round(L - usado, 3),
        })
    df_plan = pd.DataFrame(filas)

    util_global = 100.0 * total_usado / (len(plan) * L) if len(plan) else 0.0
    util_global = max(0.0, min(100.0, util_global))
    estado = "ÓPTIMO" if es_optimo else "FACTIBLE (cota superior / compactado)"
    resumen = (
        f"Método: búsqueda por factibilidad (MIN nº barras) — {estado}\n"
        f"Barras usadas: {len(plan)}\n"
        f"Utilización global: {util_global:.2f}%"
    )

    df_long = resumen_por_longitud(plan)
    df_patron = barras_por_patron(plan)

    csv_path = "plan_corte.csv"
    df_plan.to_csv(csv_path, index=False, sep=";")

    plot_df = df_plan[["Barra #", "Uso (%)"]].copy()
    return df_plan, df_long, df_patron, resumen, csv_path, plot_df

# =========================================================
# Plantilla Excel (solo Piezas)
# =========================================================

PLANTILLA_PATH = "plantilla_optim_corte.xlsx"
if not os.path.exists(PLANTILLA_PATH):
    from openpyxl import Workbook
    piezas_df = pd.DataFrame({
        "longitud_mm": [6000, 5998, 3997, 3988, 2998, 2995, 2400, 2399, 1200.5, 204.2],
        "cantidad":    [  12,   12,   18,   18,   24,   24,   16,   16,     12,    12]
    })
    with pd.ExcelWriter(PLANTILLA_PATH, engine="openpyxl") as w:
        piezas_df.to_excel(w, index=False, sheet_name="Piezas")

# =========================================================
# Handlers UI + estilo
# =========================================================

def run_manual(L, kerf, texto):
    try:
        L = float(str(L).replace(",", "."))
        kerf = float(str(kerf).replace(",", "."))
        piezas = parse_piezas_con_parentesis(texto)
        if not piezas:
            raise ValueError("No se han introducido piezas.")
        df_plan, df_long, df_patron, resumen, csv_path, plot_df = optimizar(L, piezas, kerf)
        return pd.DataFrame(), df_plan, df_long, df_patron, resumen, csv_path, plot_df
    except Exception as e:
        empty = pd.DataFrame()
        return empty, pd.DataFrame({"Error": [str(e)]}), empty, empty, f"❌ {e}", None, empty

def run_excel(file, L, kerf):
    try:
        L = float(str(L).replace(",", "."))
        kerf = float(str(kerf).replace(",", "."))
        df_pzs = leer_excel_piezas(file.name)
        piezas = piezas_desde_df(df_pzs)
        df_plan, df_long, df_patron, resumen, csv_path, plot_df = optimizar(L, piezas, kerf)
        return df_pzs, df_plan, df_long, df_patron, resumen, csv_path, plot_df
    except Exception as e:
        empty = pd.DataFrame()
        return empty, pd.DataFrame({"Error": [str(e)]}), empty, empty, f"❌ {e}", None, empty

css_custom = """
.gradio-container {max-width: 1100px !important;}
.header-card {
  background: var(--panel-background-fill);
  border: 1px solid var(--border-color-primary);
  border-radius: 16px; padding: 14px; margin-bottom: 8px;
  color: var(--body-text-color);
}
.metrics {display:flex; gap:12px; margin:8px 0 4px;}
.metric {
  flex:1; padding:12px; border-radius:14px;
  background: rgba(127,127,127,0.08); border: 1px solid var(--border-color-primary);
  text-align:center;
}
.metric b{font-size:1.1rem}
.small { font-size: 0.95rem; }
"""

with gr.Blocks(theme=gr.themes.Soft(), css=css_custom) as demo:
    gr.Markdown("## 🪚 Optimizador de corte — **mínimo nº de barras**")

    gr.HTML("""
    <div class='header-card small'>
      <p>Indica la <b>longitud de barra</b> (mm) y el <b>kerf</b> (mm). 
      Usa <b>Manual</b> o sube un <b>Excel</b> </p>
    </div>
    """)

    with gr.Row():
        L_in_top = gr.Textbox(label="Longitud de la barra (mm)", value="12000")
        kerf_in_top = gr.Textbox(label="Kerf entre piezas (mm)", value="2")

    gr.DownloadButton(label="⬇️ Descargar plantilla Excel (.xlsx)", value=PLANTILLA_PATH)

    with gr.Tabs():
        with gr.Tab("Manual"):
            piezas_in = gr.Textbox(
                label="Piezas (mm) — cantidades entre paréntesis (ej. 300(2), 400, 500(3))",
                value="6000(12), 5998(12), 3997 "
            )
            run_btn = gr.Button("Calcular plan (Manual)", variant="primary")

        with gr.Tab("Excel"):
            archivo = gr.File(label="Sube Excel (.xlsx) ", file_types=[".xlsx"])
            leer_btn = gr.Button("Leer y calcular (Excel)", variant="primary")
            prev_piezas = gr.Dataframe(label="Piezas leídas del Excel")

    # Salidas
    tabla_plan = gr.Dataframe(label="Plan de cortes por barra (ordenado)", wrap=True)
    tabla_long = gr.Dataframe(label="Resumen por longitud de corte")
    tabla_patron = gr.Dataframe(label="Barras con el mismo patrón")
    resumen_out = gr.Textbox(label="Resumen", interactive=False, lines=5)
    csv_file = gr.File(label="Descargar CSV (plan por barra)", interactive=False)
    plot = gr.BarPlot(x="Barra #", y="Uso (%)", title="Uso por barra (%)", y_lim=[0, 100])

    # Conexiones
    run_btn.click(
        run_manual,
        inputs=[L_in_top, kerf_in_top, piezas_in],
        outputs=[prev_piezas, tabla_plan, tabla_long, tabla_patron, resumen_out, csv_file, plot]
    )

    leer_btn.click(
        run_excel,
        inputs=[archivo, L_in_top, kerf_in_top],
        outputs=[prev_piezas, tabla_plan, tabla_long, tabla_patron, resumen_out, csv_file, plot]
    )

demo.launch()
# ... tu código tal cual ...
if __name__ == "__main__":
    import os
    demo.launch(server_name="0.0.0.0", server_port=int(os.environ.get("PORT", 7860)))
