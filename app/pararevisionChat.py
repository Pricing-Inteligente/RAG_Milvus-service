# -----------------------------------------------------------------------------
# /chat/stream (memoria + encabezado de filtros)
# -----------------------------------------------------------------------------
@app.post("/chat/stream", tags=["chat"])
def chat_stream(req: ChatReqStream):
    text = (req.message or "").strip()
    t_req0 = _now_ms()
    
    

        # (nuevo) Diagnóstico de "sin resultados"
    def _diagnose_no_results(intent: str, *, plan=None, text:str="", macros=None, rows=None, hits=None, agg=None) -> str:
        f = (plan.filters if plan else {}) or {}
        countries = f.get("country")
        category  = f.get("category")
        store     = f.get("store")

        # --- MACRO ---
        if intent in ("macro_list","macro_compare","macro_lookup"):
            if intent == "macro_list" and not countries:
                return "faltó indicar el país para las variables macroeconómicas. ¿Indicas uno? (p. ej., CO, MX o BR)"
            if intent in ("macro_compare","macro_lookup") and not countries:
                return "faltó el país. ¿Lo agrego? (p. ej., CO, MX o BR)"
            if rows == [] or hits == [] or (agg and not agg.get("groups")):
                return "no hay datos publicados para esa variable/país en la base actual. ¿Probamos con otra variable o país?"
            return "no encontré series para esa variable/país. ¿Quieres ver el listado completo de variables disponibles?"

        # --- PRODUCTOS ---
        # 1) categoría ausente o irreconocible
        if intent in ("lookup","list","compare","aggregate") and not category:
            cat_guess = _canonicalize_category(text)
            if not cat_guess:
                return "no identifiqué una categoría válida (p. ej., 'azúcar', 'leche líquida'). ¿Te gustaría que use una de ejemplo?"

        # 2) tienda demasiado restrictiva
        if store and ((rows == []) or (hits == []) or (agg and not agg.get("groups"))):
            return "el filtro de tienda es muy restrictivo para esa categoría/país. ¿Quito la tienda o pruebo con otra?"

        # 3) “compare” sin suficientes ítems
        if intent == "compare" and hits is not None and len(hits) < 2:
            return f"necesito al menos 2 productos comparables, pero obtuve {len(hits)}. ¿Te listo más productos o cambiamos de país/tienda?"

        # 4) “aggregate” sin grupos
        if intent == "aggregate" and (not agg or not agg.get("groups")):
            return "no hubo datos suficientes para agrupar. ¿Prefieres un promedio simple o cambiar el group_by?"

        # 5) genérico
        return "no hubo coincidencias con esa combinación de filtros. ¿Relajo filtros (sin tienda) o cambiamos de categoría/país?"

    # (nuevo) SSE de “sin datos” con razón + contexto + CTA + FIN
    def _sse_no_data_ex(reason: str, filters: dict | None):
        head = _filters_head(filters or {})
        yield f"data: Hola. No pude traer resultados porque {reason}\n\n"
        if head:
            yield f"data: {head}\n\n"
        yield "data: [FIN]\n\n"


    

    # Small-talk corto
    small = _is_smalltalk(text)
    if small:
        prompt = _prompt_smalltalk(text, small)
        _log_event("chat_stream_smalltalk", {
            "sid": req.session_id, "message": text, "intent": small
        })
        return StreamingResponse(llm_chat.stream(prompt), media_type="text/event-stream")
    

    # --- REFINAR / SET FILTROS (antes de MACRO y planner) ---
    
    if _is_refine_command(text):
        f_now = _guess_filters(text)
        if not any(f_now.values()):
            reason = "no identifiqué qué filtro cambiar (país/categoría/tienda). ¿Cuál ajusto?"
            return StreamingResponse(_sse_no_data_ex(reason, None), media_type="text/event-stream")

        # fusiona con lo último que tengamos
        last = MEM.get(req.session_id) or {}
        lastf = (last.get("last_filters") or {})
        new_filters = sanitize_filters({**lastf, **f_now})

        # guarda la memoria
        if req.session_id:
            MEM.set(req.session_id, {"last_filters": new_filters})

        # responde breve y CIERRA con [FIN]
        def gen_refine():
            yield f"data: ¡Listo! Actualicé los filtros a → {new_filters}\n\n"
            yield "data: ¿Consulto el promedio, comparo países o te muestro los más baratos?\n\n"
            yield "data: [FIN]\n\n"
        return StreamingResponse(gen_refine(), media_type="text/event-stream")


    # --- Señaladores rápidos de intent (topN, trend) antes del planner ---
    nt = _norm(text)
    force_topn = _is_topn_query(nt)
    force_trend = _is_trend_query(nt)


    
        # ------ RUTA MACRO (stream) ------
    macros = _extract_macros(text)
    if macros:
        countries = _extract_countries(text) or []
        if not countries and req.session_id:
            last = MEM.get(req.session_id) or {}
            if last.get("last_country"):
                countries = [last["last_country"]]


                # ——— SUPERLATIVOS (p.ej. "más alto/más bajo") ———
        superl = _is_macro_superlative_query(text)  # devuelve "max", "min" o None
        if superl and macros and "__ALL__" not in macros:
            var = macros[0]
            cs = _macro_default_countries()  # usa tu lista de países por defecto
            ranked = _macro_rank(var, cs)    # [{country, value, unit, date, name}, ...]

            if not ranked:
                reason = _diagnose_no_results("macro_compare", plan=None, text=text, macros=macros, rows=[])
                return StreamingResponse(_sse_no_data_ex(reason, {"country": cs}), media_type="text/event-stream")

            # ordena según max/min
            ranked.sort(key=lambda x: x["value"], reverse=(superl == "max"))
            best = ranked[0]

            def gen_super():
                yield f"data: Filtros → variable: {var} | países: {', '.join(cs)}\n\n"

                # Redacción humana
                facts = {
                    "type": "macro_rank",
                    "variable": var,
                    "order": "desc" if superl == "max" else "asc",
                    "rows": ranked[:10]
                }
                hint = f"El {'más alto' if superl=='max' else 'más bajo'} es {best['country']} ({best['value']} {best.get('unit') or ''} {best.get('date') or ''}). ¿Quieres comparar los 3 primeros?"
                prompt = _prompt_macro_humano("macro_rank", facts, hint)
                for chunk in _stream_no_fin(prompt):
                    yield chunk
                yield "data: \n\n"

                # Lista compacta
                topn = 5
                titulo = "Top 5 más altos" if superl == "max" else "Top 5 más bajos"
                yield f"data: {titulo}:\n\n"
                for i, r in enumerate(ranked[:topn], start=1):
                    yield f"data: {i}. {r['country']}: {r['value']} {r.get('unit') or ''} ({r.get('date') or ''})\n\n"

                yield "data: [FIN]\n\n"

            return StreamingResponse(gen_super(), media_type="text/event-stream")





            
                    # --- Superlativos macro: “¿qué país tiene X más alto/bajo?”
        superl = _is_macro_superlative_query(text)
        if superl and macros and "__ALL__" not in macros:
            var = macros[0]  # ya viene canónico por MACRO_ALIASES

            cs = getattr(S, "countries", None)
            if not cs:
                cs = sorted({v for v in COUNTRY_ALIASES.values() if isinstance(v, str) and len(v) == 2})

            rows = macro_compare(var, cs) or []
            if not rows:
                reason = _diagnose_no_results("macro_compare", plan=None, text=text, macros=macros, rows=rows)
                return StreamingResponse(_sse_no_data_ex(reason, {"country": cs}), media_type="text/event-stream")

            key = lambda r: float(r.get("value") or 0.0)
            best = (max(rows, key=key) if superl == "max" else min(rows, key=key))

            def gen_super():
                yield f"data: Filtros → variable: {var} | países: {' | '.join(cs)}\n\n"
                facts = {
                    "type": "macro_compare",
                    "variable": var,
                    "countries": cs,
                    "rows": [
                        {"country": x.get("country"), "name": x.get("name"),
                        "value": x.get("value"), "unit": x.get("unit"), "date": x.get("date")}
                        for x in rows
                    ],
                    "winner": {
                        "mode": "máximo" if superl == "max" else "mínimo",
                        "country": best.get("country"),
                        "value": best.get("value"),
                        "unit": best.get("unit"),
                        "date": best.get("date"),
                        "name": best.get("name"),
                    },
                }
                prompt = _prompt_macro_humano(
                    "macro_compare",
                    facts,
                    "¿Quieres que agregue otro país o ver la serie histórica?"
                )
                for chunk in _stream_no_fin(prompt):
                    yield chunk
                yield "data: [FIN]\n\n"

            return StreamingResponse(gen_super(), media_type="text/event-stream")






        # ¿hay intención de productos en este mismo turno?
        heur_now = _guess_filters(text)  # país/categoría/tienda detectados por alias
        has_products = bool(heur_now.get("category")) or bool(re.search(r"\bprecio|precios\b", _norm(text)))

        # Si hay macros + productos => MIXED
        if has_products and countries:
            def gen_mix():
                # --- Sección MACRO ---
                for m in macros:
                    if m == "__ALL__":
                        rows = macro_list(countries[0]) or []
                        if rows:
                            yield f"data: [MACRO] País: {countries[0]} | variables: TODAS (mostrando 10)\n\n"

                            facts = {
                                "type": "macro_list",
                                "country": countries[0],
                                "items": [
                                    {"name": x.get("name"), "value": x.get("value"),
                                    "unit": x.get("unit"), "date": x.get("date")}
                                    for x in (rows[:10] if rows else [])
                                ]
                            }
                            prompt = _prompt_macro_humano("macro_list", facts, "¿Quieres que me enfoque en inflación, tasa o dólar?")
                            for chunk in _stream_no_fin(prompt):
                                yield chunk
                            yield "data: \n\n"


                    elif len(countries) >= 2:
                        rows = macro_compare(m, countries) or []
                        if rows:
                            yield f"data: [MACRO] {m} | países: {' | '.join(countries)}\n\n"
                            # --- MACRO WRITER (COMPARE) ---
                            facts = {
                                "type": "macro_compare",
                                "variable": m,
                                "countries": countries,
                                "rows": [
                                    {"country": x.get("country"), "value": x.get("value"),
                                    "unit": x.get("unit"), "date": x.get("date")}
                                    for x in (rows or [])
                                ]
                            }
                            prompt = _prompt_macro_humano("macro_compare", facts, "¿Agrego otro país o convierto a misma base si aplica?")
                            for chunk in _stream_no_fin(prompt):
                                yield chunk
                            yield "data: \n\n"

                    else:
                        r = macro_lookup(m, countries[0]) if len(countries)==1 else None
                        if r:
                            yield f"data: [MACRO] País: {countries[0]} | variable: {m}\n\n"
                            # --- MACRO WRITER (LOOKUP) ---
                            facts = {
                                "type": "macro_lookup",
                                "variable": m,
                                "country": countries[0],
                                "value": (r or {}).get("value"),
                                "unit": (r or {}).get("unit"),
                                "date": (r or {}).get("date"),
                                "name": (r or {}).get("name"),
                            }
                            prompt = _prompt_macro_humano("macro_lookup", facts, "¿La comparamos con otro país o te muestro la serie?")
                            for chunk in _stream_no_fin(prompt):
                                yield chunk
                            yield "data: \n\n"


                # --- Sección PRODUCTOS ---
                pf = {k: v for k, v in (heur_now or {}).items() if k in ("country","category","store") and v}
                # si no trajo categoría exacta, igual intenta listado general por país
                rows_prod = list_by_filter(pf, limit= min(getattr(S, 'chat_list_default', 500), 40))
                if not rows_prod and pf.get("category"):
                    pf2 = dict(pf); pf2.pop("category", None)
                    rows_prod = list_by_filter(pf2, limit= min(getattr(S, 'chat_list_default', 500), 40))

                if rows_prod:
                    yield f"data: [PRODUCTOS] Filtros → {pf}\n\n"
                    for i, r in enumerate(rows_prod[:10], start=1):
                        yield f"data: {_fmt_row(r, i)}\n\n"
                yield "data: [FIN]\n\n"
            return StreamingResponse(gen_mix(), media_type="text/event-stream")

        # ---- Si NO hay productos en el mismo turno, conserva el comportamiento original ----
        try:
            if "__ALL__" in macros:
                if not countries:
                    reason = _diagnose_no_results("macro_list", plan=None, text=text, macros=macros)
                    return StreamingResponse(_sse_no_data_ex(reason, {"country": countries or None}), media_type="text/event-stream")
                
                rows = macro_list(countries[0]) or []
                if not rows:
                    reason = _diagnose_no_results("macro_list", plan=None, text=text, macros=macros, rows=rows)
                    return StreamingResponse(_sse_no_data_ex(reason, {"country": countries}), media_type="text/event-stream")

                def gen_all():
                    yield f"data: Filtros → país: {countries[0]} | variable: TODAS\n\n"
                    # --- MACRO WRITER (LISTA) ---
                    facts = {
                        "type": "macro_list",
                        "country": countries[0],
                        "items": [
                            {"name": x.get("name"), "value": x.get("value"),
                            "unit": x.get("unit"), "date": x.get("date")}
                            for x in (rows[:10] if rows else [])
                        ]
                    }
                    prompt = _prompt_macro_humano("macro_list", facts, "¿Te muestro solo inflación, tasa o dólar?")
                    for chunk in _stream_no_fin(prompt):
                        yield chunk
                    yield "data: \n\n"

                    yield f"data: Encontré {len(rows)} variable(s). Mostrando las primeras:\n\n"
                    for i, r in enumerate(rows[:10], start=1):
                        yield f"data: {_fmt_macro_row(r, i)}\n\n"
                    yield "data: [FIN]\n\n"
                MEM.set(req.session_id, {"last_country": countries[0]})
                return StreamingResponse(gen_all(), media_type="text/event-stream")

            if len(countries) >= 2:
                rows = []
                for m in macros:
                    rows.extend(macro_compare(m, countries) or [])
                if not rows:
                    reason = _diagnose_no_results("macro_compare", plan=None, text=text, macros=macros, rows=rows)
                    return StreamingResponse(_sse_no_data_ex(reason, {"country": countries}), media_type="text/event-stream")

                def gen_cmp():
                    yield f"data: Filtros → variables: {', '.join(macros)} | países: {' | '.join(countries)}\n\n"
                    # --- MACRO WRITER (COMPARE) ---
                    facts = {
                        "type": "macro_compare",
                        "variables": macros,
                        "countries": countries,
                        "rows": [
                            {"country": x.get("country"), "name": x.get("name"),
                            "value": x.get("value"), "unit": x.get("unit"), "date": x.get("date")}
                            for x in (rows or [])
                        ]
                    }
                    prompt = _prompt_macro_humano("macro_compare", facts, "¿Agrego otro país o otra variable?")
                    for chunk in _stream_no_fin(prompt):
                        yield chunk
                    yield "data: \n\n"

                    for i, r in enumerate(rows, start=1):
                        yield f"data: {_fmt_macro_row(r, i)}\n\n"
                    yield "data: [FIN]\n\n"
                return StreamingResponse(gen_cmp(), media_type="text/event-stream")

            if len(countries) == 1:
                # primera macro mencionada por simplicidad
                r = macro_lookup(macros[0], countries[0])
                if not r:
                    reason = _diagnose_no_results("macro_lookup", plan=None, text=text, macros=macros, rows=[])
                    return StreamingResponse(_sse_no_data_ex(reason, {"country": countries[0]}), media_type="text/event-stream")
                def gen_one():
                    yield f"data: Filtros → país: {countries[0]} | variable: {macros[0]}\n\n"
                    # --- MACRO WRITER (LOOKUP) ---
                    facts = {
                        "type": "macro_lookup",
                        "variable": macros[0],
                        "country": countries[0],
                        "value": (r or {}).get("value"),
                        "unit": (r or {}).get("unit"),
                        "date": (r or {}).get("date"),
                        "name": (r or {}).get("name"),
                    }
                    prompt = _prompt_macro_humano("macro_lookup", facts, "¿La comparamos con otro país o prefieres la serie histórica?")
                    for chunk in _stream_no_fin(prompt):
                        yield chunk
                    yield "data: \n\n"

                    yield f"data: {_fmt_macro_row(r)}\n\n"
                    yield "data: [FIN]\n\n"
                return StreamingResponse(gen_one(), media_type="text/event-stream")


            intent_guess = ("macro_lookup" if len(countries) == 1
                else "macro_compare" if len(countries) >= 2
                else "macro_list")
            reason = _diagnose_no_results(intent_guess, plan=None, text=text, macros=macros)
            return StreamingResponse(_sse_no_data_ex(reason, {"country": countries or None}), media_type="text/event-stream")
        except Exception as e:
            reason = f"ocurrió un error interno ({type(e).__name__}). ¿Intentamos de nuevo con filtros más simples?"
            return StreamingResponse(_sse_no_data_ex(reason, {"country": countries or None}), media_type="text/event-stream")


    # justo después del branch refine (y antes del planner)
    last = MEM.get(req.session_id) or {}
    if last and re.search(r"\b(háblame\s+de|hablame\s+de|ahora\s+en|y\s+en)\b", _norm(text or "")):
        # Si detecto “háblame de … / ahora en …”, reusar la última intención
        countries = _extract_countries(text) or []
        if countries:
            # parchar filtros en memoria
            lastf = dict(last.get("last_filters") or {})
            lastf["country"] = countries if len(countries) > 1 else countries[0]
            MEM.set(req.session_id, {"last_filters": lastf, "last_intent": last.get("last_intent")})

            # fuerza la intención anterior si es de negocio
            reuse_intent = last.get("last_intent") or "lookup"
            # crea un plan mínimo para caer en la rama correcta
            plan = Plan(
                intent=reuse_intent,
                filters=_normalize_plan_filters(merge_with_memory(lastf, req.session_id), text),
                top_k=getattr(S, "top_k", 5),
                limit=min(max(req.limit or 100, 1), getattr(S, "chat_list_max", 1000)),
            )
            # y deja que el flujo continúe con este plan (no devuelvas aquí)




        # Detectar intents especiales por texto ANTES del planner
    force_topn  = _is_topn_query(text)
    force_trend = _is_trend_query(text)

    # 1) Planner LLM (si aplica) + heurística directa  (TIMED)
    planner_ms = None                     # ← NUEVO
    t_pl0 = _now_ms()                     # ← NUEVO
    try:
        if 'plan' not in locals() or plan is None:
            plan = _plan_from_llm(text)
    finally:
        planner_ms = _now_ms() - t_pl0    # ← NUEVO

    heur_now = _guess_filters(text)  # SOLO alias explícitos del turno
    base_filters = plan.filters if plan else heur_now

    # 2) Filtros inteligentes (base + LLM + semántico si falta category)
    merged0 = build_filters_smart(text, base_filters)

    # 3) Preferir categoría anterior si NO se mencionó explícitamente una nueva
    prefer_last_cat = not bool(heur_now.get("category"))

    # 4) Fusión con memoria, señalando qué se mencionó explícitamente
    merged = merge_with_memory(
        merged0,
        req.session_id,
        prefer_last_cat=prefer_last_cat,
        mentioned={
            "category": "category" in heur_now,
            "country":  "country"  in heur_now,
            "store":    "store"    in heur_now,
        },
    )

  


    # 5) Si no hubo plan, crear uno heurístico
    if not plan:
        plan = Plan(
            intent=_classify_intent_heuristic(text),
            filters=merged,
            top_k=getattr(S, "top_k", 5),
            limit=min(max(req.limit or 100, 1), getattr(S, "chat_list_max", 1000)),
        )
    else:
        plan.filters = merged

    plan.filters = _normalize_plan_filters(plan.filters, text)

    # Si el usuario NO mencionó categoría en este turno,
# y la intención es un TOP-N o TREND, NO rellenes categoría semánticamente:
    explicit_cat = bool(heur_now.get("category"))  # heur_now ya lo calculas arriba
    if plan.intent in ("topn", "trend") and not explicit_cat:
        plan.filters.pop("category", None)  # fuerza "sin categoría" → todo el país


    if force_topn:
        plan.intent = "topn"
    elif force_trend:
        plan.intent = "trend"

    # >>> BLOQUE NUEVO: si es una consulta genérica de "precios de productos", listar por país
    def _is_generic_prices(nt: str) -> bool:
        return (("precio" in nt or "precios" in nt) and ("producto" in nt or "productos" in nt))

    nt2 = _norm(text)
    if _is_generic_prices(nt2):
        plan.intent = "list"
        # usa el limit que vino en el request si existe; si no, un default
        max_allowed = getattr(S, "chat_list_max", 1000)
        default_list = getattr(S, "chat_list_default", 500)
        plan.limit = min(max(req.limit or default_list, 1), max_allowed)
        # limpiar filtros que no pidió explícitamente
        if plan.filters:
            for k in ("category", "brand", "store"):
                plan.filters.pop(k, None)
    # <<< FIN BLOQUE NUEVO


    # Log pre-ejecución (plan + filtros)
    _log_event("chat_stream_plan", {
        "sid": req.session_id,
        "message": text,
        "plan": plan.model_dump(),
        "prefer_last_cat": prefer_last_cat,
        "explicit_mentions": {"category": "category" in heur_now,
                              "country": "country" in heur_now,
                              "store": "store" in heur_now},
    })

    # === INTENTS ===

    # ---- LIST → stream de tabla simple ----
    if plan.intent == "list":
        try:
            rows = list_by_filter(
                plan.filters or None,limit = min(max(plan.limit or 100, 1), getattr(S, "chat_list_max", 1000)))
            
            if not rows and plan.filters and plan.filters.get("category"):
                f2 = dict(plan.filters); f2.pop("category", None)
                rows = list_by_filter(f2, limit=min(max(plan.limit or 100, 1), getattr(S, "chat_list_max", 1000)))
        except Exception as e:
            _log_event("chat_stream_list_error", {"sid": req.session_id, "err": str(e)[:200]})
            reason = f"ocurrió un error interno ({type(e).__name__})."
            return StreamingResponse(_sse_no_data_ex(reason, plan.filters), media_type="text/event-stream")


        remember_session(req.session_id, filters=plan.filters, intent="list", query=text, hits=rows)
        _log_event("chat_stream_list", {
            "sid": req.session_id,
            "filters": plan.filters,
            "count": len(rows),
            "sample_ids": [r.get("product_id") for r in (rows[:10] or [])],
        })
        if not rows:
            reason = _diagnose_no_results("list", plan=plan, text=text, rows=rows)
            return StreamingResponse(_sse_no_data_ex(reason, plan.filters), media_type="text/event-stream")

        def gen():
            # --- VIZ_PROMPT ---
            try:
                vizp = _maybe_viz_prompt("list", plan.filters or {}, rows=rows)
            except NameError:
                vizp = None
            if vizp:
                yield f"data: [VIZ_PROMPT] {vizp}\n\n"

            yield f"data: {_filters_head(plan.filters)}\n\n"
            # Mini-writer para saludo + contexto + 1-2 hallazgos + CTA
            try:
                sample = [
                    {
                        "name": r.get("name"), "brand": r.get("brand"),
                        "price": r.get("price"), "currency": r.get("currency"),
                        "store": r.get("store")
                    } for r in rows[:5]
                ]
                prompt_summary = (
                    "Eres el asistente del SPI. Responde con: saludo breve → contexto "
                    "(país/categoría si están) → breve resumen de hallazgos (menciona 1–2 ejemplos) "
                    "→ CTA único (p.ej., \"¿Te muestro solo los más baratos por tienda?\").\n"
                    f"Contexto: filtros={json.dumps(plan.filters or {}, ensure_ascii=False)}\n"
                    f"Ejemplos(JSON): {json.dumps(sample, ensure_ascii=False)}"
                )
                summary_txt = llm_chat.generate(prompt_summary).strip()
                if summary_txt:
                    yield f"data: {summary_txt}\n\n"
            except Exception:
                pass

            yield f"data: Encontré {len(rows)} producto(s). Mostrando los primeros 10:\n\n"
            for i, r in enumerate(rows[:10], start=1):
                line = (
                    f"{i}. {r.get('name')} · Marca: {r.get('brand')} · "
                    f"Pres: {r.get('size')}{r.get('unit')} · "
                    f"Precio: {r.get('price')} {r.get('currency')} · "
                    f"Tienda: {r.get('store')} · País: {r.get('country')} "
                    f"[{r.get('product_id')}]"
                )
                yield f"data: {line}\n\n"

            yield "data: Sugerencia: ¿quieres ver solo los más baratos por tienda o filtrar por marca?\n\n"   
            yield "data: [FIN]\n\n"
        return StreamingResponse(gen(), media_type="text/event-stream",
                                 headers={"Cache-Control": "no-cache", "Connection": "keep-alive"})

    

        # ---- TOPN → top N más baratos/caros ----
    if plan.intent == "topn":
        try:
            n, mode = _extract_topn(text)
            # trae una muestra amplia y ordena en memoria
            rows = list_by_filter(
                plan.filters or None,
                limit=min(max(plan.limit or getattr(S, "chat_list_default", 500), 1), getattr(S, "chat_list_max", 1000))
            ) or []
        except Exception as e:
            reason = f"ocurrió un error interno ({type(e).__name__})."
            return StreamingResponse(_sse_no_data_ex(reason, plan.filters), media_type="text/event-stream")

        if not rows:
            reason = _diagnose_no_results("list", plan=plan, text=text, rows=rows)
            return StreamingResponse(_sse_no_data_ex(reason, plan.filters), media_type="text/event-stream")

        rows = [r for r in rows if r.get("price") is not None]
        rows.sort(key=lambda r: float(r["price"]), reverse=(mode != "cheap"))
        top = rows[:n]

        def gen_topn():
            # VIZ opcional (barras topN)
            try:
                vizp = _maybe_viz_prompt("topn", plan.filters or {}, rows=top)
            except NameError:
                vizp = None
            if vizp:
                yield f"data: [VIZ_PROMPT] {vizp}\n\n"

            yield f"data: {_filters_head(plan.filters)}\n\n"

            # Mini-writer humano (no stream ≠ evita doble FIN)
            try:
                sample = [
                    {"name": r.get("name"), "brand": r.get("brand"), "price": r.get("price"),
                     "currency": r.get("currency"), "store": r.get("store")}
                    for r in top[:3]
                ]
                prompt_summary = (
                    "Eres el asistente del SPI. Formato: saludo breve → contexto (país/categoría/tienda si están) "
                    "→ resumen del TOP con 1–2 ejemplos → CTA único (p.ej., \"¿Filtramos por tienda o marca?\").\n"
                    f"Contexto: filtros={json.dumps(plan.filters or {}, ensure_ascii=False)}, n={n}, modo={mode}\n"
                    f"Ejemplos(JSON): {json.dumps(sample, ensure_ascii=False)}"
                )
                txt = llm_chat.generate(prompt_summary).strip()
                if txt:
                    yield f"data: {txt}\n\n"
            except Exception:
                pass

            yield f"data: TOP {n} {'más baratos' if mode=='cheap' else 'más caros'}:\n\n"
            for i, r in enumerate(top, 1):
                yield f"data: {_fmt_row(r, i)}\n\n"
            yield "data: [FIN]\n\n"

        return StreamingResponse(gen_topn(), media_type="text/event-stream",
                                 headers={"Cache-Control": "no-cache", "Connection": "keep-alive"})


       
     # ---- TREND → tendencia últimos 30 días (o rango corto) ----
    if plan.intent == "trend":
        try:
            # puedes extraer días del texto si quieres; por ahora 30
            days = 30
            ser = series_prices(plan.filters or None, days=days) or []
        except Exception as e:
            reason = f"ocurrió un error interno ({type(e).__name__})."
            return StreamingResponse(_sse_no_data_ex(reason, plan.filters), media_type="text/event-stream")

        if not ser:
            reason = _diagnose_no_results("aggregate", plan=plan, text=text, agg={"groups": []})
            return StreamingResponse(_sse_no_data_ex(reason, plan.filters), media_type="text/event-stream")

        vals = [s.get("value") for s in ser if s.get("value") is not None]
        if len(vals) >= 2 and vals[0]:
            pct = (vals[-1] - vals[0]) / vals[0] * 100.0
        else:
            pct = 0.0

        facts = {
            "days": days,
            "pct": pct,
            "last": vals[-1] if vals else None,
            "currency": ser[-1].get("currency"),
            "n": len(vals),
            "date_start": ser[0].get("date"),
            "date_end": ser[-1].get("date"),
            "filters": plan.filters or {}
        }

        prompt = (
            "Eres el asistente del SPI. Responde con: saludo breve → contexto (país/categoría, rango de fechas) "
            "→ tendencia con % y último valor/fecha → CTA único (p.ej., \"¿Genero la gráfica o comparo otro país?\").\n"
            f"FACTS(JSON): {json.dumps(facts, ensure_ascii=False)}\n"
            "RESPUESTA:"
        )

        def gen_trend():
            # VIZ_PROMPT: línea temporal
            try:
                vizp = _maybe_viz_prompt("trend", plan.filters or {}, series=ser)
            except NameError:
                vizp = None
            if vizp:
                yield f"data: [VIZ_PROMPT] {vizp}\n\n"

            yield f"data: {_filters_head(plan.filters)}\n\n"

            # Writer en stream (usa _stream_no_fin para evitar doble FIN)
            for chunk in _stream_no_fin(prompt):
                yield chunk
            yield "data: [FIN]\n\n"

        return StreamingResponse(gen_trend(), media_type="text/event-stream",
                                 headers={"Cache-Control": "no-cache", "Connection": "keep-alive"})



    # ---- COMPARE → N países (2..10) usando SOLO columnas de la BD ----
    if plan.intent == "compare":
        try:
            countries = _extract_countries(text, max_n=10)
            cat = (plan.filters or {}).get("category")

            # --- NUEVO: inferir categoría si falta (sinónimos -> semántico) ---
            if not cat:
                cat = _canonicalize_category(text)
                
                if cat:
                    plan.filters = dict(plan.filters or {}, category=cat)

            # Si el usuario mencionó >=2 países y (ahora sí) hay categoría → multi-compare
            if len(countries) >= 2 and cat:
                per_country_rows: list[tuple[str, list[dict]]] = []
                top_per_country = max(min(plan.top_k or 3, 5), 1)  # 1..5 por país

                # No arrastramos tienda al comparar países
                for code in countries:
                    f = dict(plan.filters or {})
                    f["country"] = code
                    f.pop("store", None)
                    rows = list_by_filter(
                        f, limit=min(max(plan.limit or 100, 1), 1000)
                    ) or []
                    per_country_rows.append((code, rows))

                with_data = [(c, rows) for c, rows in per_country_rows if rows]
                without_data = [c for c, rows in per_country_rows if not rows]

                _log_event("chat_stream_compare_multi", {
                    "sid": req.session_id,
                    "category": cat,
                    "countries": countries,
                    "with_data": {c: len(rows) for c, rows in with_data},
                    "without_data": without_data[:10],
                    "samples": {c: [r.get("product_id") for r in rows[:3]] for c, rows in with_data},
                })

                # Política: necesitamos al menos 2 países con datos
                if len(with_data) < 2:
                    try:
                        suggestions = {}
                        for code in without_data:
                            # agregados por categoría en ese país (sin forzar "cafe")
                            agg_cat = aggregate_prices({"country": code}, by="category") or {}
                            groups = agg_cat.get("groups") or []
                            # prioriza categorías que contengan "cafe" (normalizado)
                            def _n(s): return (s or "").lower()
                            candidates = [g.get("group") for g in groups if g and g.get("group")]
                            cafe_like = [c for c in candidates if "cafe" in _n(c) or "caf" in _n(c) or "coffee" in _n(c)]
                            suggestions[code] = cafe_like[:3] or candidates[:3]  # top 3
                    except Exception:
                        suggestions = {}

                    reason = f"necesito al menos 2 países con datos para '{cat}', pero tuve " \
                            f"{len(with_data)} con datos y {len(without_data)} sin datos."
                    def gen_hint():
                        yield f"data: Hola. No pude comparar porque {reason}\n\n"
                        yield f"data: Filtros → países: {countries} | categoría: {cat}\n\n"
                        for code in without_data:
                            opts = suggestions.get(code) or []
                            if opts:
                                yield f"data: Sugerencia para {code}: prueba con categoría(s) {', '.join(opts)}\n\n"
                            else:
                                yield f"data: Sugerencia para {code}: prueba sin categoría o con otra similar.\n\n"
                        yield "data: [FIN]\n\n"
                    return StreamingResponse(gen_hint(), media_type="text/event-stream")


                # --- Si quieres redacción humana, usa LLM con "hechos" agregados
                if getattr(S, "compare_llm", True):
                    # preparar hechos por país
                    facts = {"category": cat, "countries": []}
                    for c, rows in with_data:
                        prices = [r.get("price") for r in rows if r.get("price") is not None]
                        if not prices:
                            continue
                        # moneda más común
                        from collections import Counter
                        cur = None
                        if rows:
                            cur = Counter([r.get("currency") for r in rows if r.get("currency")]).most_common(1)[0][0]
                        brands = {r.get("brand") for r in rows if r.get("brand")}
                        facts["countries"].append({
                            "country": c,
                            "avg": sum(prices) / max(len(prices), 1),
                            "min": min(prices),
                            "max": max(prices),
                            "n": len(prices),
                            "brands_n": len(brands),
                            "currency": cur
                        })

                    ctx_json = json.dumps(facts, ensure_ascii=False)
                    prompt = (
                        f"Eres el asistente del Sistema Pricing Inteligente (SPI). "
                        f"Redacta de forma natural y amable una comparativa de precios para la categoría '{cat}'. "
                        f"Usa exclusivamente estos HECHOS (JSON): {ctx_json}. "
                        "Estructura: saludo breve → contexto → respuesta comparativa (promedio, mínimo, máximo y conteo de marcas por país) → "
                        "cierre con un mini resumen y un call to action para filtrar más."
                    )

                    def gen():
                        # VIZ opcional
                        try:
                            groups = [
                                {"group": d["country"], "avg": d["avg"], "min": d["min"], "max": d["max"]}
                                for d in facts["countries"]
                            ]
                            vizp = _maybe_viz_prompt(
                                "aggregate",
                                {"category": cat, "country": [d["country"] for d in facts["countries"]]},
                                agg={"groups": groups},
                                group_by="country",
                            )
                        except NameError:
                            vizp = None
                        if vizp:
                            yield f"data: [VIZ_PROMPT] {vizp}\n\n"

                        head = " | ".join(countries)
                        yield f"data: Filtros → categoría: {cat} | países: {head} | tienda: -\n\n"

                        # LLM stream
                        t_llm0 = _now_ms(); first_token_ms = None
                        for chunk in _stream_no_fin(prompt):
                            if first_token_ms is None:
                                first_token_ms = _now_ms()
                            yield chunk
                        _log_perf("chat_stream_compare_llm_perf", {
                            "gen_model": llm_chat.model,
                            "ttfb_ms": (first_token_ms - t_llm0) if first_token_ms else None,
                            "sid": req.session_id, "q": text[:120]
                        })
                        yield "data: [FIN]\n\n"
                        

                    return StreamingResponse(
                        gen(), media_type="text/event-stream",
                        headers={"Cache-Control": "no-cache", "Connection": "keep-alive"}
                    )

                # --- Respaldo determinista (sin LLM)
                def gen():
                    try:
                        groups = []
                        for c, rows in with_data:
                            prices = [r.get("price") for r in rows if r.get("price") is not None]
                            if not prices:
                                continue
                            groups.append({
                                "group": c,
                                "avg": sum(prices) / max(len(prices), 1),
                                "min": min(prices),
                                "max": max(prices),
                            })
                        vizp = _maybe_viz_prompt(
                            "aggregate",
                            {"category": cat, "country": [c for c, _ in with_data]},
                            agg={"groups": groups},
                            group_by="country",
                        )
                    except NameError:
                        vizp = None
                    if vizp:
                        yield f"data: [VIZ_PROMPT] {vizp}\n\n"

                    head = " | ".join(countries)
                    yield f"data: Filtros → categoría: {cat} | países: {head} | tienda: -\n\n"
                    if without_data:
                        yield f"data: Aviso: sin registros para: {', '.join(without_data)}\n\n"
                    for c, rows in with_data:
                        yield f"data: — País {c}: mostrando hasta {top_per_country} producto(s)\n\n"
                        for i, r in enumerate(rows[:top_per_country], start=1):
                            yield f"data: {_fmt_row(r, i)}\n\n"
                    yield "data: [FIN]\n\n"

                return StreamingResponse(
                    gen(), media_type="text/event-stream",
                    headers={"Cache-Control": "no-cache", "Connection": "keep-alive"}
                )

            # --- Fallback: comparación simple cuando no hay (N países + categoría) ---
            effective_q = pick_effective_query(
                text, req.session_id, prefer_last_cat=not bool((plan.filters or {}).get("category"))
            )
            _log_event("chat_stream_plan", {
                "sid": req.session_id,
                "message": text,
                "plan": plan.model_dump() if plan else None,
                "merged_filters": merged
            })
            hits = retrieve(effective_q, plan.filters or None)[: max(plan.top_k or 5, 5)]
            _log_event("chat_stream_compare_single", {
                "sid": req.session_id,
                "filters": plan.filters,
                "effective_query": effective_q,
                "ids": [h.get("product_id") for h in (hits or [])],
            })
            if len(hits) < 2:
                reason = _diagnose_no_results("compare", plan=plan, text=text, hits=hits)
                return StreamingResponse(_sse_no_data_ex(reason, plan.filters), media_type="text/event-stream")

            def gen_single():
                try:
                    vizp = _maybe_viz_prompt("compare", plan.filters or {}, rows=hits[:2])
                except NameError:
                    vizp = None
                if vizp:
                    yield f"data: [VIZ_PROMPT] {vizp}\n\n"
                yield f"data: Filtros → país: {(plan.filters or {}).get('country') or '-'} | categoría: {(plan.filters or {}).get('category') or '-'} | tienda: {(plan.filters or {}).get('store') or '-'}\n\n"
                # --- Mini-writer humano (no stream) para saludo + contexto + mini-comparación + CTA ---
                try:
                    sample = [
                        {
                            "name": h.get("name"),
                            "brand": h.get("brand"),
                            "price": h.get("price"),
                            "currency": h.get("currency"),
                            "store": h.get("store"),
                            "country": h.get("country")
                        } for h in hits[:2]
                    ]
                    prompt_summary = (
                        "Eres el asistente del SPI. Responde con: saludo breve → contexto "
                        "(categoría/país/tienda si están) → mini-comparación clara de los 2 resultados "
                        "→ CTA único (p.ej., \"¿Quieres que lo ordene o convertir a la misma moneda?\").\n"
                        f"Contexto: filtros={json.dumps(plan.filters or {}, ensure_ascii=False)}\n"
                        f"Ejemplos(JSON): {json.dumps(sample, ensure_ascii=False)}"
                    )
                    summary_txt = llm_chat.generate(prompt_summary).strip()
                    if summary_txt:
                        yield f"data: {summary_txt}\n\n"
                except Exception:
                    pass

                yield "data: Comparativa simple (primeros 2 resultados):\n\n"
                for i, h in enumerate(hits[:2], start=1):
                    yield f"data: {_fmt_row(h, i)}\n\n"
                yield "data: [FIN]\n\n"

            return StreamingResponse(gen_single(), media_type="text/event-stream")

        except Exception as e:
            _log_event("chat_stream_compare_error", {"sid": req.session_id, "err": str(e)[:200]})
            reason = f"ocurrió un error interno ({type(e).__name__})."
            return StreamingResponse(_sse_no_data_ex(reason, plan.filters), media_type="text/event-stream")


    # ---- AGGREGATE → stream de resumen agregado ----
    if plan.intent == "aggregate":
        try:
            t_agg0 = _now_ms()
            agg = aggregate_prices(plan.filters or None, by=plan.group_by or "category")
            t_agg1 = _now_ms()
        except Exception as e:
            _log_event("chat_stream_aggregate_error", {"sid": req.session_id, "err": str(e)[:200]})
            reason = f"ocurrió un error interno ({type(e).__name__})."
            return StreamingResponse(_sse_no_data_ex(reason, plan.filters), media_type="text/event-stream")

        remember_session(req.session_id, filters=plan.filters, intent="aggregate", query=text, hits=[])
        _log_event("chat_stream_aggregate", {
            "sid": req.session_id,
            "filters": plan.filters,
            "group_by": plan.group_by or "category",
            "rows": agg.get("groups", [])[:10],
        })
        if not agg.get("groups"):
            reason = _diagnose_no_results("aggregate", plan=plan, text=text, agg=agg)
            return StreamingResponse(_sse_no_data_ex(reason, plan.filters), media_type="text/event-stream")

        ctx = json.dumps({"group_by": plan.group_by or "category",
                        "rows": agg.get("groups", [])}, ensure_ascii=False)
        prompt = (
                    f"Eres {ASSISTANT_NAME}. "
                    "Redacta SIEMPRE en 2–4 frases con: saludo breve → contexto (group_by y filtros) "
                    "→ resumen de cifras visibles en el JSON → CTA único (p.ej., "
                    "\"¿Genero una gráfica o convierto a una moneda común?\"). "
                    f"Usa SOLO este JSON: {ctx}. No inventes valores."
                    )


        def gen():
            # VIZ_PROMPT (no cuenta para TTFB del LLM)
            try:
                vizp = _maybe_viz_prompt("aggregate", plan.filters or {}, agg=agg, group_by=plan.group_by or "category")
            except NameError:
                vizp = None
            if vizp:
                yield f"data: [VIZ_PROMPT] {vizp}\n\n"

            yield f"data: {_filters_head(plan.filters)}\n\n"

            # ---- LLM STREAM con TTFB y duración total ----
            t_llm0 = _now_ms()
            first_token_ms = None
            total_chars = 0
            for chunk in _stream_no_fin(prompt):
                if first_token_ms is None:
                    first_token_ms = _now_ms()
                total_chars += len(chunk)
                yield chunk
            t_llm1 = _now_ms()

            _log_perf("chat_stream_aggregate_perf", {
                "gen_model": llm_chat.model,
                "planner_ms": planner_ms,
                "ttfb_ms": (first_token_ms - t_llm0) if first_token_ms else None,
                "llm_stream_ms": t_llm1 - t_llm0,
                "aggregate_ms": t_agg1 - t_agg0,
                "total_ms": _now_ms() - t_req0,
                "rows": len(agg.get("groups", [])),
                "filters": plan.filters,
                "q": text[:120],
                "sid": req.session_id
            })
            yield "data: [FIN]\n\n"


        return StreamingResponse(gen(), media_type="text/event-stream")


   # ---- LOOKUP (por defecto) → stream respuesta amable + contexto ----
    try:
        t_ret0 = _now_ms()
        effective_q = pick_effective_query(text, req.session_id, prefer_last_cat)

        # Empezamos con los filtros planeados
        facts_filters = dict(plan.filters or {})

        hits = retrieve(effective_q, facts_filters)[: plan.top_k or getattr(S, "top_k", 5)]

        # Fallback: si no hay hits y había categoría, reintenta sin categoría
        if not hits and facts_filters.get("category"):
            f2 = dict(facts_filters); f2.pop("category", None)
            h2 = retrieve(effective_q, f2)[: plan.top_k or getattr(S, "top_k", 5)]
            if h2:
                hits = h2
                facts_filters = f2   # <-- ¡clave! usa estos filtros para calcular facts

        t_ret1 = _now_ms()


        # Si los hits muestran una categoría dominante, úsala para facts
        from collections import Counter
        hit_cats = [h.get("category") for h in hits if h.get("category")]
        if hit_cats:
            top_cat = Counter(hit_cats).most_common(1)[0][0]
            if top_cat:
                facts_filters["category"] = top_cat


    except Exception as e:
        _log_event("chat_stream_lookup_error", {"sid": req.session_id, "err": str(e)[:200]})
        reason = f"ocurrió un error interno ({type(e).__name__})."
        return StreamingResponse(_sse_no_data_ex(reason, plan.filters), media_type="text/event-stream")

    remember_session(req.session_id, filters=plan.filters, intent="lookup", query=text, hits=hits)
    _log_event("chat_stream_lookup", {
        "sid": req.session_id,
        "filters": plan.filters,
        "effective_query": effective_q,
        "ids": [h.get("product_id") for h in (hits or [])],
    })

    if not hits:
        reason = _diagnose_no_results("lookup", plan=plan, text=text, hits=hits)

        return StreamingResponse(_sse_no_data_ex(reason, plan.filters), media_type="text/event-stream")

    ctx = _build_ctx(hits, plan.top_k or getattr(S, "top_k", 5))

    # --- HECHOS desde la base para que el LLM redacte ---
    # --- HECHOS desde la base para que el LLM redacte ---
    base_filters = dict(facts_filters)
    base_filters.pop("store", None)  # promedio nacional sin tienda

    # 1) Promedio nacional (base de BD)
    rows_all = list_by_filter(base_filters, limit=min(getattr(S, "aggregate_limit", 5000), 5000))
    prices = [r.get("price") for r in rows_all if r.get("price") is not None]

    cur = None
    if rows_all:
        from collections import Counter
        cur = Counter([r.get("currency") for r in rows_all if r.get("currency")]).most_common(1)[0][0]

    # Fallback: si no hubo filas para avg pero sí hay hits, calcula desde hits
    if not prices and hits:
        prices = [h.get("price") for h in hits if h.get("price") is not None]
        if prices and not cur:
            from collections import Counter
            cur = Counter([h.get("currency") for h in hits if h.get("currency")]).most_common(1)[0][0]

    avg_all = (sum(prices) / max(len(prices), 1)) if prices else None

    # 2) Promedios por marca (BD)
    agg_brand = aggregate_prices(base_filters, by="brand")
    groups = agg_brand.get("groups") or []
    groups = [g for g in groups if g and g.get("group") not in (None, "", "N/A") and g.get("avg") is not None]

    # Fallback: si no hay grupos en BD, calcula desde los hits
    if not groups and hits:
        tmp_sum, tmp_n = {}, {}
        for h in hits:
            b, p = (h.get("brand") or "N/A"), h.get("price")
            if p is None: continue
            tmp_sum[b] = tmp_sum.get(b, 0.0) + float(p)
            tmp_n[b] = tmp_n.get(b, 0) + 1
        groups = [{"group": b, "avg": tmp_sum[b]/max(tmp_n[b],1), "n": tmp_n[b]} for b in tmp_sum]

    # Orden y selección top-N
    groups.sort(key=lambda g: float(g.get("avg") or 0.0), reverse=True)
    max_lines = int(getattr(S, "brand_avg_max", 8))
    brands = []
    for g in groups[:max_lines]:
        brands.append({
            "brand": str(g.get("group")),
            "avg": float(g.get("avg")),
            "n": int(g.get("n") or 0),
        })

    # Rango min–máx para el resumen final
    brand_range = None
    if brands:
        lo = min(brands, key=lambda b: b["avg"])
        hi = max(brands, key=lambda b: b["avg"])
        brand_range = {
            "min_brand": lo["brand"], "min_avg": lo["avg"],
            "max_brand": hi["brand"], "max_avg": hi["avg"],
        }

    facts = {
        "country": (plan.filters or {}).get("country"),
        "category": facts_filters.get("category") or (plan.filters or {}).get("category"),
        "currency": cur,
        "national_avg": float(avg_all) if avg_all is not None else None,
        "n": len(prices),
        "brands": brands,
        "brand_range": brand_range,
    }


    prompt = _prompt_lookup_from_facts(text, facts, ctx)

    def gen_lookup():
        # VIZ_PROMPT + encabezado (no cuentan para TTFB del LLM)
        try:
            vizp = _maybe_viz_prompt("lookup", plan.filters or {}, rows=hits)
        except NameError:
            vizp = None
        if vizp:
            yield f"data: [VIZ_PROMPT] {vizp}\n\n"

        yield f"data: {_filters_head(plan.filters)}\n\n"

        # ---- LLM STREAM con TTFB y duración total ----
        
        t_llm0 = _now_ms()
        first_token_ms = None
        total_chars = 0
        for chunk in _stream_no_fin(prompt):
            if first_token_ms is None:
                first_token_ms = _now_ms()
            total_chars += len(chunk)
            yield chunk
        t_llm1 = _now_ms()

        _log_perf("chat_stream_lookup_perf", {
            "gen_model": llm_chat.model,
            "planner_ms": planner_ms,
            "retrieve_ms": t_ret1 - t_ret0,
            "ttfb_ms": (first_token_ms - t_llm0) if first_token_ms else None,
            "llm_stream_ms": t_llm1 - t_llm0,
            "total_ms": _now_ms() - t_req0,
            "hits": len(hits),
            "ctx_len_chars": len(ctx),
            "top_k": plan.top_k or getattr(S, "top_k", 5),
            "filters": plan.filters,
            "q": text[:120],
            "sid": req.session_id,
        })

        yield "data: [FIN]\n\n"

    return StreamingResponse(gen_lookup(), media_type="text/event-stream")