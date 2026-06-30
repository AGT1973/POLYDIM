"""
POLYDIM — Ejercicio para Alumnos #02
=====================================
Segundo nivel: META y autoprogramacion entre IAs.
Se ejecuta directamente:

    python3 polydim_alumno_ejercicio_02.py

Requiere haber completado el Ejercicio #01 (Space, ObjectND, conexion basica).
Cada paso imprime lo que esta pasando y verifica con assert que el
resultado es el esperado.

Requiere: polydim_runtime_v04.py en el mismo directorio (o instalado).
Dependencias: numpy

Estructura del ejercicio (5 pasos):
    PASO 1 — Conectar dos IAs con META_PERMISSIONS distintos (negociacion)
    PASO 2 — Definir un handler y reprogramar una IA remota con REPROGRAM
    PASO 3 — Consultar el estado interno de una IA remota con QUERY
    PASO 4 — Seguridad: un permiso no negociado se ignora silenciosamente
    PASO 5 — META_DEPTH_LIMIT: por que no se permite autoprogramacion encadenada

Al final: un mini-desafio que el alumno debe completar solo,
con un assert oculto que valida si lo resolvio correctamente.

Autor: Claude Sonnet 4.6
Version: V0.1 — 2026-06-16 — TASK_026
Base: polydim_runtime_v04.py (Session con META_PERMISSIONS y META_DEPTH_LIMIT)
"""

from polydim_runtime_v04 import (
    Space, ObjectND, Session,
    META_ACK, META_REPROGRAM, META_RESET, META_QUERY,
)


def linea():
    print("─" * 64)


def paso(n, titulo):
    print()
    linea()
    print(f"PASO {n}: {titulo}")
    linea()


# ─────────────────────────────────────────────
# PASO 1
# ─────────────────────────────────────────────

def handler_eco(obj, ctx, session):
    """Handler de ejemplo: no transforma nada, solo existe para ser detectado."""
    return []


def paso_1():
    paso(1, "Conectar dos IAs con META_PERMISSIONS distintos")

    print("""
Cada Session declara que tipos de instrucciones META esta dispuesta
a ACEPTAR (no a enviar). Cuando dos IAs se conectan, el handshake()
negocia la INTERSECCION de esos permisos: solo lo que ambas declararon
queda habilitado para esa sesion.

IA_X va a poder pedir ACK, REPROGRAM y QUERY.
IA_Y acepta ACK, REPROGRAM, RESET y QUERY (RESET es exclusivo de IA_Y).
""")

    sp_x = Space("ALUMNO_X")
    sp_y = Space("ALUMNO_Y")

    ses_x = Session(sp_x, "IA_X",
                     meta_permissions=[META_ACK, META_REPROGRAM, META_QUERY])
    ses_y = Session(sp_y, "IA_Y",
                     meta_permissions=[META_ACK, META_REPROGRAM, META_RESET, META_QUERY],
                     handlers={"eco": handler_eco})

    print(">>> ses_x = Session(sp_x, 'IA_X', meta_permissions=[META_ACK, META_REPROGRAM, META_QUERY])")
    print(">>> ses_y = Session(sp_y, 'IA_Y', meta_permissions=[META_ACK, META_REPROGRAM, META_RESET, META_QUERY], handlers={'eco': handler_eco})")

    ses_x.connect(ses_y)
    print("\n>>> ses_x.connect(ses_y)")

    negociados = ses_y.negotiated_meta_perms
    print(f">>> ses_y.negotiated_meta_perms -> {sorted(negociados)}")

    assert META_REPROGRAM in negociados, "META_REPROGRAM deberia quedar negociado (ambas lo declaran)"
    assert META_QUERY in negociados, "META_QUERY deberia quedar negociado (ambas lo declaran)"
    assert META_RESET not in negociados, (
        "META_RESET NO deberia quedar negociado: IA_X nunca lo declaro, "
        "asi que la interseccion lo excluye aunque IA_Y si lo acepte."
    )
    print("\n[OK] La negociacion es una interseccion real: lo que una IA no ofrece, no queda habilitado.")
    return ses_x, ses_y


# ─────────────────────────────────────────────
# PASO 2
# ─────────────────────────────────────────────

def paso_2(ses_x, ses_y):
    paso(2, "Reprogramar IA_Y a distancia con DIM_META + REPROGRAM")

    print("""
Una instruccion META no es codigo arbitrario: es un ObjectND mas, con
una dimension DIM_META activa. El receptor la interpreta como una
orden solo si tiene el permiso correspondiente ya negociado.

tipo='REPROGRAM' le dice a la IA receptora: "cambia tu handler activo
al que tengas registrado bajo este nombre".
""")

    instruccion = ObjectND(ses_x.space).add(
        "DIM_META", {"tipo": "REPROGRAM", "nuevo_modo": "eco"}, w=1.0
    )
    print(">>> instruccion = ObjectND(ses_x.space).add('DIM_META', {'tipo': 'REPROGRAM', 'nuevo_modo': 'eco'}, w=1.0)")

    ses_y.receive(ses_x.send(instruccion))
    print(">>> ses_y.receive(ses_x.send(instruccion))")

    print(f"\n>>> ses_y.handler -> {ses_y.handler}")
    print(f">>> ses_y.ctx -> {ses_y.ctx}")

    assert ses_y.handler is handler_eco, (
        "ses_y.handler deberia ser handler_eco tras el REPROGRAM. "
        "Si esto falla, revisa que 'eco' este en el dict handlers de ses_y."
    )
    assert "reprogramada_en" in ses_y.ctx, "el contexto deberia registrar en que seq se reprogramo"
    print("\n[OK] IA_Y cambio su propio comportamiento a partir de un mensaje, no de codigo local.")
    return instruccion


# ─────────────────────────────────────────────
# PASO 3
# ─────────────────────────────────────────────

def paso_3(ses_x, ses_y):
    paso(3, "Consultar el estado interno de IA_Y con QUERY")

    print("""
QUERY no modifica nada: pide un dato del estado interno de la sesion
receptora (que handler tiene activo, en que modo esta, etc.) y lo
deja guardado en su propio ctx bajo la clave '_query_<campo>'.
""")

    consulta = ObjectND(ses_x.space).add(
        "DIM_META", {"tipo": "QUERY", "campo": "handler"}, w=1.0
    )
    print(">>> consulta = ObjectND(ses_x.space).add('DIM_META', {'tipo': 'QUERY', 'campo': 'handler'}, w=1.0)")

    ses_y.receive(ses_x.send(consulta))
    print(">>> ses_y.receive(ses_x.send(consulta))")

    respuesta = ses_y.ctx.get("_query_handler")
    print(f"\n>>> ses_y.ctx['_query_handler'] -> {respuesta!r}")

    assert respuesta == "handler_eco", (
        f"se esperaba 'handler_eco' (el nombre del handler activo tras el PASO 2), "
        f"pero se obtuvo {respuesta!r}"
    )
    print("\n[OK] IA_X pudo inspeccionar el estado de IA_Y sin tocar su memoria directamente.")


# ─────────────────────────────────────────────
# PASO 4
# ─────────────────────────────────────────────

def paso_4(ses_x, ses_y):
    paso(4, "Seguridad: un permiso no negociado se ignora en silencio")

    print("""
IA_Y SI declaro META_RESET en su propia lista. Pero la negociacion es
de a dos: como IA_X nunca lo declaro, META_RESET no quedo en la
interseccion negociada de esta sesion. El protocolo no devuelve un
error: simplemente ignora la instruccion. Esto es deliberado — evita
que una IA mal configurada genere ruido de errores por cada permiso
que no comparte.
""")

    handler_antes = ses_y.handler
    intento_reset = ObjectND(ses_x.space).add(
        "DIM_META", {"tipo": "RESET", "nivel": "handler"}, w=1.0
    )
    print(">>> intento_reset = ObjectND(ses_x.space).add('DIM_META', {'tipo': 'RESET', 'nivel': 'handler'}, w=1.0)")

    ses_y.receive(ses_x.send(intento_reset))
    print(">>> ses_y.receive(ses_x.send(intento_reset))")

    print(f"\n>>> ses_y.handler is handler_antes -> {ses_y.handler is handler_antes}")

    assert ses_y.handler is handler_antes, (
        "ses_y.handler no deberia haber cambiado: META_RESET nunca quedo "
        "negociado en esta sesion, asi que la instruccion debio ignorarse."
    )
    print("\n[OK] Permiso no negociado = instruccion ignorada. IA_Y sigue reprogramada como 'eco'.")


# ─────────────────────────────────────────────
# PASO 5
# ─────────────────────────────────────────────

def paso_5(ses_x, ses_y):
    paso(5, "META_DEPTH_LIMIT: por que no se permite autoprogramacion encadenada")

    print("""
Si un REPROGRAM pudiera, a su vez, disparar otro REPROGRAM, una IA
podria entrar en un bucle de autoprogramacion sin control. Por eso
existe META_DEPTH_LIMIT=1: mientras una sesion YA esta procesando una
instruccion META, ignora cualquier instruccion META adicional que
llegue en ese mismo instante.

Simulamos esa situacion fijando manualmente _meta_depth en el limite
(en la practica, esto lo hace el propio receive() durante el
procesamiento; aca lo forzamos para poder observarlo).
""")

    ses_y._meta_depth = 1
    print(">>> ses_y._meta_depth = 1   # simula 'ya estoy procesando un META'")

    otro_intento = ObjectND(ses_x.space).add(
        "DIM_META", {"tipo": "REPROGRAM", "nuevo_modo": "otro_modo_inexistente"}, w=1.0
    )
    ses_y.receive(ses_x.send(otro_intento))
    print(">>> ses_y.receive(ses_x.send(otro_intento))")

    print(f"\n>>> ses_y.handler is handler_eco -> {ses_y.handler is handler_eco}")

    assert ses_y.handler is handler_eco, (
        "ses_y.handler no deberia haber cambiado: con _meta_depth en el limite, "
        "receive() ni siquiera deberia haber llamado a _procesar_meta()."
    )

    ses_y._meta_depth = 0
    print(">>> ses_y._meta_depth = 0   # volvemos al estado normal")
    print("\n[OK] El limite de profundidad bloquea la cadena antes de que empiece.")


# ─────────────────────────────────────────────
# MINI-DESAFIO
# ─────────────────────────────────────────────

def desafio():
    paso("DESAFIO", "Reprograma una IA vos mismo")

    print("""
Te toca a vos. Completa la funcion `resolver()` de abajo:

    1. Define una funcion handler propia, por ejemplo:
           def modo_silencio(obj, ctx, session):
               return []

    2. Crea dos Spaces y dos Sessions nuevas. La sesion RECEPTORA debe
       registrar tu handler en su dict `handlers`, con el nombre que
       vos elijas (por ejemplo "silencio"). Ambas sesiones deben
       declarar META_REPROGRAM en su meta_permissions para que la
       negociacion lo habilite.

    3. Conecta las sesiones con .connect().

    4. Envia un ObjectND con DIM_META {'tipo': 'REPROGRAM', 'nuevo_modo': '<tu_nombre>'}
       desde la sesion emisora hacia la receptora.

    5. Devolve la tupla (sesion_receptora, tu_funcion_handler) desde resolver().

No hay solucion impresa aca a proposito: el assert de abajo verifica
tu resultado sin revelarte el camino.
""")

    def resolver():
        # ── Escribi tu solucion aca abajo ──────────────────────────
        #
        # sp_a = Space(...)
        # sp_b = Space(...)
        # def mi_handler(obj, ctx, session):
        #     return []
        # ses_a = Session(...)
        # ses_b = Session(..., handlers={...})
        # ses_a.connect(ses_b)
        # orden = ObjectND(ses_a.space).add("DIM_META", {...}, w=1.0)
        # ses_b.receive(ses_a.send(orden))
        # return ses_b, mi_handler
        raise NotImplementedError("Completa resolver() para intentar el desafio")

    try:
        sesion_receptora, mi_handler = resolver()
    except NotImplementedError:
        print("[PENDIENTE] Todavia no completaste resolver(). Editá la funcion y volve a correr el script.")
        return

    assert sesion_receptora.handler is mi_handler, (
        "La sesion receptora no quedo reprogramada con tu handler. "
        "Revisa que el nombre en 'nuevo_modo' coincida con la clave en el dict handlers, "
        "y que META_REPROGRAM haya quedado negociado entre ambas sesiones."
    )
    print("\n[OK] Resolviste el desafio: reprogramaste una IA remota con tu propio handler.")


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────

if __name__ == "__main__":
    ses_x, ses_y = paso_1()
    paso_2(ses_x, ses_y)
    paso_3(ses_x, ses_y)
    paso_4(ses_x, ses_y)
    paso_5(ses_x, ses_y)
    desafio()

    print()
    linea()
    print("Ejercicio #02 completo. Ahora sabes negociar permisos, reprogramar")
    print("una IA remota, consultar su estado y por que existe un limite de")
    print("profundidad para la autoprogramacion.")
    linea()
