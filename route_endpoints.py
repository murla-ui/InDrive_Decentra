import math
import numpy as np
import pandas as pd
import networkx as nx

from pathlib import Path
import sys
import argparse

# ---- Гео-хелперы ----
def _meters_xy(lat, lng, ref_lat=None):

    lat = np.asarray(lat, dtype=float)
    lng = np.asarray(lng, dtype=float)
    if ref_lat is None or np.isnan(ref_lat):
        ref_lat = np.nanmean(lat)
    m_per_deg_lat = 111_320.0
    m_per_deg_lng = 111_320.0 * math.cos(math.radians(ref_lat))
    x = (lng - np.nanmean(lng)) * m_per_deg_lng
    y = (lat - np.nanmean(lat)) * m_per_deg_lat
    return x, y

def _bearing_deg_from_xy(x0, y0, x1, y1):
    # 0° = север, по часовой
    ang = math.degrees(math.atan2((x1 - x0), (y1 - y0)))
    return ang + 360.0 if ang < 0 else ang

def _circ_diff_deg(a, b):
    if a is None or b is None or np.isnan(a) or np.isnan(b):
        return np.nan
    d = abs(a - b) % 360.0
    return min(d, 360.0 - d)

# ---- Граф: kNN → MST → диаметр ----
def _knn_graph(X, k):
    """X: [n,2] в метрах; возвращает связный граф крупнейшей компоненты."""
    n = X.shape[0]
    G = nx.Graph()
    if n == 0:
        return G
    if n == 1:
        G.add_node(0)
        return G

    D = np.sqrt(((X[:, None, :] - X[None, :, :])**2).sum(axis=2))
    np.fill_diagonal(D, np.inf)
    kk = min(k, n - 1)

    for i in range(n):
        G.add_node(i)
        nn_idx = np.argpartition(D[i], kk)[:kk]
        for j in nn_idx:
            w = float(D[i, j])
            if np.isfinite(w):
                G.add_edge(i, j, weight=w)

    if G.number_of_edges() == 0:
        for i in range(n):
            for j in range(i + 1, n):
                G.add_edge(i, j, weight=float(D[i, j]))

    if not nx.is_connected(G):
        comps = sorted(nx.connected_components(G), key=len, reverse=True)
        G = G.subgraph(comps[0]).copy()

    return G

def _mst_diameter_endpoints(G):
    """Возвращает MST и индексы двух самых удалённых узлов (диаметр по кратчайшему пути)."""
    T = nx.minimum_spanning_tree(G, weight="weight")
    if T.number_of_nodes() == 1:
        node = next(iter(T.nodes()))
        return T, node, node
    leaves = [u for u, d in T.degree() if d == 1] or list(T.nodes())
    src = leaves[0]
    d1 = nx.single_source_dijkstra_path_length(T, src, weight="weight")
    a = max(d1, key=d1.get)
    d2 = nx.single_source_dijkstra_path_length(T, a, weight="weight")
    b = max(d2, key=d2.get)
    return T, a, b

# ---- Хребет (A↔B) и криволинейная абсцисса ----
def _spine_and_abscissa(X, path_nodes):
    """
    Возвращает:
      spine_pts (m), cum_path (массив кумулятивной длины), s_all (s-коорд. для всех точек), on_spine (bool)
    """
    P = X[path_nodes]
    segs = P[1:] - P[:-1]
    seglen = np.sqrt((segs**2).sum(axis=1))
    cum = np.concatenate([[0.0], np.cumsum(seglen)])

    s_on_path = {node: float(cum[i]) for i, node in enumerate(path_nodes)}
    n = X.shape[0]
    s_all = np.empty(n, dtype=float)
    on_spine = np.zeros(n, dtype=bool)

    seg_dirs = segs
    seg_l2 = (seg_dirs**2).sum(axis=1)

    for i in range(n):
        if i in s_on_path:
            s_all[i] = s_on_path[i]
            on_spine[i] = True
            continue
        q = X[i]
        best_dist = np.inf
        best_s = 0.0
        for j in range(len(segs)):
            A = P[j]
            v = seg_dirs[j]
            l2 = seg_l2[j]
            if l2 == 0.0:
                proj = A
                d = np.linalg.norm(q - proj)
                s = cum[j]
            else:
                t = float(np.dot(q - A, v) / l2)
                t = 0.0 if t < 0 else (1.0 if t > 1 else t)
                proj = A + t * v
                d = np.linalg.norm(q - proj)
                s = cum[j] + t * math.sqrt(l2)
            if d < best_dist:
                best_dist = d
                best_s = s
        s_all[i] = best_s

    return P, cum, s_all, on_spine

def _pick_start_end_with_azimuth(X, path_nodes, azm_arr=None, spd_arr=None):
    """Различить Start/End по азимуту (fallback: скорость). Возвращает индексы start, end, метод, уверенность."""
    if len(path_nodes) == 0:
        return 0, 0, "none", np.nan
    a = path_nodes[0]
    b = path_nodes[-1]
    a2 = path_nodes[1] if len(path_nodes) >= 2 else a
    b2 = path_nodes[-2] if len(path_nodes) >= 2 else b

    bearing_A = _bearing_deg_from_xy(*X[a], *X[a2])
    bearing_B = _bearing_deg_from_xy(*X[b], *X[b2])
    diff_A = _circ_diff_deg(azm_arr[a] if azm_arr is not None else np.nan, bearing_A)
    diff_B = _circ_diff_deg(azm_arr[b] if azm_arr is not None else np.nan, bearing_B)

    method = "azm"
    conf = np.nan
    if (np.isnan(diff_A) and np.isnan(diff_B)) and (spd_arr is not None):
        method = "speed"
        sA = spd_arr[a] if not np.isnan(spd_arr[a]) else np.inf
        sB = spd_arr[b] if not np.isnan(spd_arr[b]) else np.inf
        start, end = (a, b) if sA <= sB else (b, a)
        conf = abs(sA - sB)
    else:
        if np.isnan(diff_A): diff_A = 180.0
        if np.isnan(diff_B): diff_B = 180.0
        start, end = (a, b) if diff_A <= diff_B else (b, a)
        conf = abs(diff_B - diff_A)

    return start, end, method, conf

# ---- Основное: реконструкция маршрутов и концов ----
def reconstruct_routes(
    df,
    group_col="randomized_id",
    lat_col="lat",
    lng_col="lng",
    spd_col="spd",
    azm_col="azm",
    k=5,
):
    """
    Возвращает:
      endpoints_df: по одной строке на маршрут (A/B + Start/End + метод и уверенность)
      ordered_df: исходные точки маршрутов с 's_m', 'on_spine', 'is_start','is_end'
    """
    endpoints_rows = []
    ordered_parts = []

    for rid, g in df.groupby(group_col, sort=False):
        g = g.copy().dropna(subset=[lat_col, lng_col])
        if len(g) == 0:
            continue

        lat = g[lat_col].to_numpy(dtype=float)
        lng = g[lng_col].to_numpy(dtype=float)
        X = np.stack(_meters_xy(lat, lng, ref_lat=np.nanmean(lat)), axis=1)

        # 1) kNN-граф на исходных индексах 0..n-1
        G0 = _knn_graph(X, k=k)
        kept_idx = np.array(sorted(G0.nodes()), dtype=int)

        # 2) согласуем данные и перенумеруем узлы в 0..m-1
        mapping = {old: new for new, old in enumerate(kept_idx)}
        G = nx.relabel_nodes(G0, mapping, copy=True)

        X = X[kept_idx]
        lat = lat[kept_idx]
        lng = lng[kept_idx]
        g = g.iloc[kept_idx].reset_index(drop=True)

        if G.number_of_nodes() == 0:
            continue

        if G.number_of_nodes() == 1:
            one = g.iloc[0]
            endpoints_rows.append({
                group_col: rid,
                "A_lat": float(one[lat_col]), "A_lng": float(one[lng_col]),
                "B_lat": float(one[lat_col]), "B_lng": float(one[lng_col]),
                "start_lat": float(one[lat_col]), "start_lng": float(one[lng_lng]) if 'lng_lng' in g.columns else float(one[lng_col]),
                "end_lat":   float(one[lat_col]), "end_lng":   float(one[lng_col]),
                "method": "degenerate", "direction_confidence": 1.0, "n_points_used": 1
            })
            one_out = g.copy()
            one_out["s_m"] = 0.0
            one_out["on_spine"] = True
            one_out["is_start"] = True
            one_out["is_end"] = True
            one_out[group_col] = rid
            ordered_parts.append(one_out)
            continue

        # 3) MST → диаметр → путь A↔B
        T, a_node, b_node = _mst_diameter_endpoints(G)
        path_nodes = list(map(int, nx.shortest_path(T, a_node, b_node, weight="weight")))

        # 4) Хребет и s-координаты
        spine_pts, cum_path, s_all, on_spine = _spine_and_abscissa(X, path_nodes)

        # 5) Start/End
        azm_arr = g[azm_col].to_numpy(dtype=float) if azm_col in g.columns else None
        spd_arr = g[spd_col].to_numpy(dtype=float) if spd_col in g.columns else None
        start_i, end_i, method, conf = _pick_start_end_with_azimuth(X, path_nodes, azm_arr, spd_arr)

        # 6) Собираем выход
        g["s_m"] = s_all
        g["on_spine"] = on_spine
        g["is_start"] = False
        g["is_end"] = False
        g.loc[start_i, "is_start"] = True
        g.loc[end_i,   "is_end"]   = True

        g_sorted = g.sort_values("s_m", kind="mergesort").reset_index(drop=True)
        g_sorted[group_col] = rid
        ordered_parts.append(g_sorted)

        endpoints_rows.append({
            group_col: rid,
            "A_lat": float(lat[path_nodes[0]]),  "A_lng": float(lng[path_nodes[0]]),
            "B_lat": float(lat[path_nodes[-1]]), "B_lng": float(lng[path_nodes[-1]]),
            "start_lat": float(g.loc[start_i, lat_col]), "start_lng": float(g.loc[start_i, lng_col]),
            "end_lat":   float(g.loc[end_i,   lat_col]), "end_lng":   float(g.loc[end_i,   lng_col]),
            "method": method, "direction_confidence": float(conf) if conf is not None else np.nan,
            "n_points_used": len(g_sorted)
        })

    endpoints_df = pd.DataFrame(endpoints_rows)
    ordered_df = pd.concat(ordered_parts, ignore_index=True) if ordered_parts else pd.DataFrame()
    return endpoints_df, ordered_df

# ---- Итоговая таблица концов (уникально на маршрут) ----
def make_unique_endpoints_table(
    df,
    group_col="randomized_id",
    lat_col="lat",
    lng_col="lng",
    spd_col="spd",
    azm_col="azm",
    k=5,
    round_to=None,
):
    endpoints_df, _ = reconstruct_routes(
        df, group_col=group_col, lat_col=lat_col, lng_col=lng_col,
        spd_col=spd_col, azm_col=azm_col, k=k
    )

    if endpoints_df.empty:
        return pd.DataFrame(columns=[group_col, "start_lat", "start_lng", "end_lat", "end_lng"])

    # при дублях берём вариант с макс. числом использованных точек
    if endpoints_df.duplicated(subset=[group_col]).any():
        idx = endpoints_df.groupby(group_col)["n_points_used"].idxmax()
        endpoints_df = endpoints_df.loc[idx]

    out = endpoints_df[[group_col, "start_lat", "start_lng", "end_lat", "end_lng"]].copy()
    out = out.drop_duplicates(subset=[group_col]).reset_index(drop=True)

    if round_to is not None:
        for c in ["start_lat", "start_lng", "end_lat", "end_lng"]:
            out[c] = out[c].round(round_to)

    assert out[group_col].is_unique
    return out

# ---- Heatmap стартов/финишей по Астане (Folium) ----
def plot_endpoints_heatmap(
    endpoints_unique: pd.DataFrame,
    filename: str = "astana_endpoints_heatmap.html",
    zoom_start: int = 11,
    radius: int = 22,
    blur: int = 18,
    min_opacity: float = 0.25,
    start_gradient: dict = None,
    end_gradient: dict = None,
    show_layer_control: bool = True,
):
    """
    Отрисовывает два HeatMap-слоя: Starts (A) и Ends (B).
    Требуются пакеты: folium, folium.plugins.HeatMap
    """
    import folium
    from folium.plugins import HeatMap

    # Брендовые градиенты по умолчанию (можно поменять)
    if start_gradient is None:
        start_gradient = {0.2: "#E5BA83", 0.5: "#E59752", 0.8: "#D64550", 1.0: "#BB4A4A"}  # peach → red
    if end_gradient is None:
        end_gradient   = {0.2: "#376C8A", 0.5: "#28738A", 0.8: "#168980", 1.0: "#0F5C55"}  # deep teal

    # центр Астаны (фикс)
    center_lat, center_lng = 51.169392, 71.449074

    # подготовка и агрегация (одинаковые координаты → вес)
    starts = endpoints_unique[["start_lat", "start_lng"]].dropna()
    ends   = endpoints_unique[["end_lat", "end_lng"]].dropna()

    def to_weighted(df2):
        if df2.empty:
            return []
        g = df2.groupby(df2.columns.tolist()).size().reset_index(name="weight")
        return g.values.tolist()  # [lat, lng, weight]

    start_data = to_weighted(starts)
    end_data   = to_weighted(ends)

    m = folium.Map(location=[center_lat, center_lng], zoom_start=zoom_start, tiles="OpenStreetMap")

    if start_data:
        fg_s = folium.FeatureGroup(name="Start — Heatmap")
        HeatMap(start_data, radius=radius, blur=blur, min_opacity=min_opacity, gradient=start_gradient).add_to(fg_s)
        fg_s.add_to(m)
    if end_data:
        fg_e = folium.FeatureGroup(name="Finish — Heatmap")
        HeatMap(end_data, radius=radius, blur=blur, min_opacity=min_opacity, gradient=end_gradient).add_to(fg_e)
        fg_e.add_to(m)

    if show_layer_control:
        folium.LayerControl(collapsed=False).add_to(m)

    m.save(filename)
    return filename

# ---- CLI ----
def _build_arg_parser():
    p = argparse.ArgumentParser(
        description="Extract route start/end points per randomized_id and optionally create a heatmap (Astana)."
    )
    p.add_argument("csv_path", help="Path to input CSV with columns randomized_id, lat, lng (spd, azm optional).")
    p.add_argument("--out-csv", help="Path to save endpoints table CSV. Default: <input>_endpoints_unique.csv")
    p.add_argument("--out-html", help="Path to save heatmap HTML. Default: <input>_endpoints_heatmap.html")
    p.add_argument("--no-heatmap", action="store_true", help="Do not generate heatmap HTML.")
    p.add_argument("--k", type=int, default=5, help="k for kNN graph (default: 5).")
    p.add_argument("--round", type=int, default=6, help="Round coordinates to N decimals (default: 6).")
    # column overrides
    p.add_argument("--id-col", default="randomized_id")
    p.add_argument("--lat-col", default="lat")
    p.add_argument("--lng-col", default="lng")
    p.add_argument("--spd-col", default="spd")
    p.add_argument("--azm-col", default="azm")
    # heatmap params
    p.add_argument("--zoom-start", type=int, default=11)
    p.add_argument("--radius", type=int, default=22)
    p.add_argument("--blur", type=int, default=18)
    p.add_argument("--min-opacity", type=float, default=0.25)
    return p

def main():
    parser = _build_arg_parser()
    args = parser.parse_args()

    in_path = Path(args.csv_path)
    if not in_path.exists():
        print(f"[ERROR] File not found: {in_path}", file=sys.stderr)
        sys.exit(1)

    # defaults for outputs
    out_csv = Path(args.out_csv) if args.out_csv else in_path.with_name(in_path.stem + "_endpoints_unique.csv")
    out_html = Path(args.out_html) if args.out_html else in_path.with_name(in_path.stem + "_endpoints_heatmap.html")

    # load CSV
    df = pd.read_csv(in_path)

    # sanity check cols
    missing = [c for c in [args.id_col, args.lat_col, args.lng_col] if c not in df.columns]
    if missing:
        print(f"[ERROR] Missing required columns: {missing}", file=sys.stderr)
        sys.exit(1)

    # compute endpoints unique table
    endpoints_unique = make_unique_endpoints_table(
        df,
        group_col=args.id_col,
        lat_col=args.lat_col,
        lng_col=args.lng_col,
        spd_col=args.spd_col if args.spd_col in df.columns else "spd",
        azm_col=args.azm_col if args.azm_col in df.columns else "azm",
        k=args.k,
        round_to=args.round,
    )
    endpoints_unique.to_csv(out_csv, index=False)
    print(f"[OK] Saved endpoints table: {out_csv}  (rows={len(endpoints_unique)})")

    # heatmap
    if not args.no_heatmap:
        try:
            html_path = plot_endpoints_heatmap(
                endpoints_unique,
                filename=str(out_html),
                zoom_start=args.zoom_start,
                radius=args.radius,
                blur=args.blur,
                min_opacity=args.min_opacity,
            )
            print(f"[OK] Saved heatmap HTML: {html_path}")
        except Exception as e:
            print(f"[WARN] Heatmap failed: {e}  (you can disable with --no-heatmap)")

if __name__ == "__main__":
    main()