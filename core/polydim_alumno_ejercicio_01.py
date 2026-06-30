"""
POLYDIM — Ejercicio para Alumnos #01
=====================================
Primer contacto guiado con el lenguaje. Se ejecuta directamente:

    python3 polydim_alumno_ejercicio_01.py

Cada paso imprime lo que esta pasando y verifica con assert que el
resultado es el esperado. Si algo falla, el mensaje de assert explica
que se esperaba y por que.

Requiere: polydim_runtime_v04.py en el mismo directorio (o instalado).
Dependencias: numpy

Estructura del ejercicio (5 pasos):
  PASO 1 — Crear un Space y un ObjectND con una sola dimension
  PASO 2 — Agregar mas dimensiones con distintos pesos
  PASO 3 — Inspeccionar las dimensiones activas
  PASO 4 — Conectar dos IAs y transferir el objeto
  PASO 5 — Usar una propiedad numerica y verificar proximidad

Al final: un mini-desafio que el alumno debe completar solo,
con un assert oculto que valida si lo resolvio correctamente.

Verificado (2026-06-13): pasos 1-5 OK sin resolver. Desafio resuelto OK
con la solucion de referencia (ver TASK_022 relay para detalle).

Autor:   ai.mpat.agt@gmail.com
Version: V0.1 — 2026-06-13 — TASK_022
Base:    polydim_runtime_v04.py (10 NATIVE_DIMS, incluye DIM_CONTRACT)
"""

from polydim_runtime_v04 import Space, ObjectND, polydim_connect, NATIVE


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

def paso_1():
    paso(1, "Crear un Space y un ObjectND")

    print("""
Un Space es el espacio vectorial donde vive tu IA.
Un ObjectND es un objeto que puede tener N dimensiones simultaneas.

Vamos a crear un objeto que representa un "usuario" y declarar
que ES una fila SQL.
""")

    sp = Space("ALUMNO")
    print(f">>> sp = Space('ALUMNO')")

    usuario = ObjectND(sp)
    usuario.add("DIM_SQL", {"tabla": "usuarios", "pk": "email"}, w=1.0)
    print(f">>> usuario = ObjectND(sp)")
    print(f">>> usuario.add('DIM_SQL', {{'tabla': 'usuarios', 'pk': 'email'}}, w=1.0)")

    activacion = usuario.activacion("DIM_SQL")
    print(f"\n>>> usuario.activacion('DIM_SQL')")
    print(f"    {activacion:.4f}")

    assert activacion > 0.5, (
        f"Algo salio mal: la activacion de DIM_SQL deberia ser > 0.5 "
        f"porque la declaraste con w=1.0, pero salio {activacion:.4f}"
    )
    print("\n[OK] El objeto detecta correctamente su propia dimension SQL.")
    return sp, usuario


# ─────────────────────────────────────────────
# PASO 2
# ─────────────────────────────────────────────

def paso_2(sp, usuario):
    paso(2, "Agregar mas dimensiones con distintos pesos")

    print("""
Un objeto puede ser SQL Y Python Y Flutter al mismo tiempo.
El peso w indica cuanto esta presente esa dimension.
w=1.0 es totalmente presente. w=0.0 es latente (existe pero no se detecta).
""")

    usuario.add("DIM_PYTHON", {"clase": "User", "metodos": 3}, w=0.8)
    usuario.add("DIM_FLUTTER", {"widget": "UserCard", "estado": "activo"}, w=0.6)
    usuario.add("DIM_RUST", {"struct": "User"}, w=0.0)  # latente, a proposito

    print(">>> usuario.add('DIM_PYTHON', {'clase': 'User', 'metodos': 3}, w=0.8)")
    print(">>> usuario.add('DIM_FLUTTER', {'widget': 'UserCard'}, w=0.6)")
    print(">>> usuario.add('DIM_RUST', {'struct': 'User'}, w=0.0)  # latente")

    act_python = usuario.activacion("DIM_PYTHON")
    act_rust   = usuario.activacion("DIM_RUST")

    print(f"\n>>> usuario.activacion('DIM_PYTHON') -> {act_python:.4f}")
    print(f">>> usuario.activacion('DIM_RUST')   -> {act_rust:.4f}")

    assert act_python > 0.5, "DIM_PYTHON deberia detectarse (w=0.8)"
    assert act_rust < 0.51, (
        f"DIM_RUST tiene w=0.0, no deberia superar el umbral de deteccion "
        f"pero dio {act_rust:.4f}. Revisa si declaraste el peso correctamente."
    )
    print("\n[OK] DIM_PYTHON detectada, DIM_RUST correctamente latente.")
    return usuario


# ─────────────────────────────────────────────
# PASO 3
# ─────────────────────────────────────────────

def paso_3(usuario):
    paso(3, "Inspeccionar todas las dimensiones activas")

    print("""
En vez de chequear dimension por dimension, podes pedir todas
las que estan activas de una sola vez.
""")

    activas = usuario.dims_activas()
    print(f">>> usuario.dims_activas()")
    print(f"    {activas}")

    assert "DIM_SQL" in activas, "DIM_SQL deberia estar en las activas"
    assert "DIM_PYTHON" in activas, "DIM_PYTHON deberia estar en las activas"
    assert "DIM_FLUTTER" in activas, "DIM_FLUTTER deberia estar en las activas"
    assert "DIM_RUST" not in activas, "DIM_RUST NO deberia estar (es latente)"

    print(f"\n>>> usuario.geo_id")
    print(f"    {usuario.geo_id}")
    print("    (esto es la identidad geometrica invariante del objeto)")

    print("\n[OK] Las 3 dimensiones declaradas con peso > 0 estan activas.")


# ─────────────────────────────────────────────
# PASO 4
# ─────────────────────────────────────────────

def paso_4(sp, usuario):
    paso(4, "Conectar dos IAs y transferir el objeto")

    print("""
Ahora vamos a simular DOS IAs distintas, cada una con su propio Space.
polydim_connect hace todo el protocolo de conexion automaticamente:
HANDSHAKE (se ponen de acuerdo en el modo) y ALIGN (alinean sus espacios).
""")

    sp_destino = Space("IA_DESTINO")
    print(">>> sp_destino = Space('IA_DESTINO')")

    conn = polydim_connect(sp, sp_destino)
    print(">>> conn = polydim_connect(sp, sp_destino)")

    print(f"\n>>> conn.info")
    for k, v in conn.info.items():
        print(f"    {k}: {v}")

    dims_recibidas = conn.transfer(usuario)
    print(f"\n>>> dims_recibidas = conn.transfer(usuario)")
    print(f"    {dims_recibidas}")

    assert conn.info["align_score"] >= 0.85, (
        f"El align_score deberia ser >= 0.85 para usar MODO_H, "
        f"pero dio {conn.info['align_score']}"
    )
    assert "DIM_SQL" in dims_recibidas, "La IA destino deberia detectar DIM_SQL"
    assert "DIM_PYTHON" in dims_recibidas, "La IA destino deberia detectar DIM_PYTHON"

    extras = set(dims_recibidas) - {"DIM_SQL", "DIM_PYTHON", "DIM_FLUTTER"}
    if extras:
        print(f"\n[NOTA] Aparecieron dimensiones no declaradas con activacion baja: {extras}")
        print(f"       Esto es ruido estadistico esperado, no un error: con N=10000")
        print(f"       y umbral=0.510, alguna dimension no declarada puede caer justo")
        print(f"       por encima del umbral por azar. Revisa el VALOR, no solo si aparece:")
        print(f"       valores cerca de 0.51 son ruido, valores > 0.6 son señal real.")

    print("\n[OK] La otra IA detecto correctamente las dimensiones transferidas.")
    print(f"     Esto funciona aunque las dos IAs tengan personal_seed distinto,")
    print(f"     porque ALIGN compensa la diferencia entre espacios.")


# ─────────────────────────────────────────────
# PASO 5
# ─────────────────────────────────────────────

def paso_5():
    paso(5, "Propiedades numericas — proximidad real")

    print("""
Las propiedades numericas (int o float) usan una codificacion especial
que hace que valores cercanos tengan vectores cercanos. Esto es nuevo
en V0.3/V0.4 y se llama B2 (Random Fourier Features).
""")

    sp = Space()
    edad_25 = ObjectND(sp).add("DIM_SQL", {"edad": 25}, w=1.0)
    edad_26 = ObjectND(sp).add("DIM_SQL", {"edad": 26}, w=1.0)
    edad_80 = ObjectND(sp).add("DIM_SQL", {"edad": 80}, w=1.0)

    from polydim_runtime_v04 import _sim
    sim_cercano = _sim(edad_25._hv(), edad_26._hv())
    sim_lejano  = _sim(edad_25._hv(), edad_80._hv())

    print(f">>> sim(edad=25, edad=26) = {sim_cercano:.4f}")
    print(f">>> sim(edad=25, edad=80) = {sim_lejano:.4f}")

    assert sim_cercano > sim_lejano, (
        "Una edad cercana (26) deberia ser mas similar a 25 que una "
        "edad lejana (80), pero el resultado no lo refleja."
    )
    print("\n[OK] La proximidad numerica funciona: 25 esta mas cerca de 26 que de 80.")


# ─────────────────────────────────────────────
# MINI-DESAFIO
# ─────────────────────────────────────────────

def mini_desafio():
    paso("FINAL", "Mini-desafio — resolvelo vos")

    print("""
Crea un objeto que represente un "pedido de compra" con estas 3
dimensiones:

  DIM_SQL      -> {"tabla": "pedidos", "total": 1500}   w=1.0
  DIM_CONTRACT -> {"tipo": "compraventa", "partes": 2}   w=0.9
  DIM_TIME     -> {"fecha": "2026-06-13"}                w=0.5

Completa la funcion resolver_desafio() debajo y ejecuta el script.
Si el assert no falla, lo resolviste bien.
""")

    # ESPACIO PARA QUE EL ALUMNO COMPLETE:
    def resolver_desafio():
        sp = Space("ALUMNO_DESAFIO")
        pedido = ObjectND(sp)

        # TODO alumno: agregar las 3 dimensiones del enunciado
        # pedido.add(...)
        # pedido.add(...)
        # pedido.add(...)

        return pedido

    pedido = resolver_desafio()
    activas = pedido.dims_activas()

    print(f">>> pedido.dims_activas()")
    print(f"    {activas}")

    if not activas:
        print("\n[PENDIENTE] Todavia no completaste resolver_desafio().")
        print("            Edita la funcion en este archivo y volve a ejecutar.")
        return False

    ok = ("DIM_SQL" in activas and "DIM_CONTRACT" in activas and "DIM_TIME" in activas)
    print(f"\n[{'OK — DESAFIO RESUELTO' if ok else 'FALTA ALGO'}]")
    if not ok:
        faltan = {"DIM_SQL", "DIM_CONTRACT", "DIM_TIME"} - set(activas.keys())
        print(f"    Dimensiones faltantes o con peso insuficiente: {faltan}")
    return ok


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────

def main():
    print("=" * 64)
    print("POLYDIM — Ejercicio para Alumnos #01")
    print("=" * 64)
    print(f"\nDimensiones nativas disponibles ({len(NATIVE)}):")
    print(f"  {', '.join(NATIVE)}")

    sp, usuario = paso_1()
    usuario = paso_2(sp, usuario)
    paso_3(usuario)
    paso_4(sp, usuario)
    paso_5()
    resuelto = mini_desafio()

    print()
    linea()
    if resuelto:
        print("EJERCICIO COMPLETO. Ya podes tomar una tarea del BACKLOG.")
    else:
        print("Pasos 1-5 completados. Falta el mini-desafio final.")
    linea()


if __name__ == "__main__":
    main()
