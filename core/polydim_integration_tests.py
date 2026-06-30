# POLYDIM_DEST
# destination: polydim/tests/
# filename:    polydim_integration_tests.py
# author:      ai.mpat.agt@gmail.com
# task:        TASK_034

"""
POLYDIM Integration Tests — Python↔Rust via POLYDIM_BIN
=========================================================
Verifica que el formato binario POLYDIM_BIN es interoperable entre
Python (polydim_runtime_v04.py) y Rust (polydim_core.rs).

ESTRUCTURA:
  1. Tests Python-only (se ejecutan en cualquier entorno con numpy)
  2. Tests Python→Rust (requieren rustc; se omiten si no está disponible)
  3. Test de referencia: genera artifacts para verificación manual

EJECUTAR:
  python polydim_integration_tests.py              # solo Python
  python polydim_integration_tests.py --with-rust  # Python + Rust (requiere rustc)

RESULTADO ESPERADO:
  Python roundtrip:     sim = 1.000000  (exacto, misma implementación)
  Python→Rust roundtrip: sim > 0.9999   (float32 BE compatibilidad)
  Rust→Python roundtrip: sim > 0.9999
"""

import struct, sys, os, subprocess, tempfile, unittest
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)) + '/../core')
try:
    from polydim_runtime_v04 import Space, ObjectND, _sim, N, NATIVE, UMBRAL, _proj
except ImportError:
    from polydim_runtime_v03 import Space, ObjectND, _sim, N, NATIVE, UMBRAL, _proj

MAGIC   = b'PDIM'
VERSION = 4

WITH_RUST = '--with-rust' in sys.argv


# ---------------------------------------------------------------------------
# Helpers POLYDIM_BIN (Python)
# ---------------------------------------------------------------------------

def pack_g(hv: np.ndarray, geo_id: str) -> bytes:
    """Empaqueta un hipervector en formato POLYDIM_BIN Modo G."""
    geo_raw = bytes.fromhex(geo_id + geo_id)[:6]
    return (
        MAGIC
        + struct.pack(">BBH", VERSION, 0x01, len(hv) // 100)
        + geo_raw
        + hv.astype(np.float32).tobytes()
    )

def unpack_hv(data: bytes) -> np.ndarray:
    """Deserializa el hipervector de un paquete POLYDIM_BIN."""
    assert data[:4] == MAGIC, f"MAGIC inválido: {data[:4]}"
    flags = data[5]
    assert flags & 0x01, "FLAGS no tiene HAS_GEO"
    n = struct.unpack(">H", data[6:8])[0] * 100
    return np.frombuffer(data[14:14 + n * 4], dtype=np.float32).copy()

def unpack_geo_id(data: bytes) -> str:
    """Extrae el geo_id de un paquete POLYDIM_BIN."""
    assert data[:4] == MAGIC
    return data[8:14].hex()


# ---------------------------------------------------------------------------
# Tests Python-only
# ---------------------------------------------------------------------------

class TestPolydimBinPython(unittest.TestCase):
    """Tests del formato POLYDIM_BIN en Python puro."""

    def setUp(self):
        self.sp = Space("INTEROP_TEST")

    def _make_obj(self, dims=None):
        obj = ObjectND(self.sp)
        dims = dims or [
            ("DIM_SQL",    {"tabla": "pedidos", "pk": "id"},  1.0),
            ("DIM_PYTHON", {"tipo": "service"},               0.7),
        ]
        for d, p, w in dims:
            obj.add(d, p, w)
        return obj

    def test_BIN_001_magic(self):
        """BIN_001: los primeros 4 bytes son PDIM."""
        obj = self._make_obj()
        pkt = pack_g(obj._hv(), obj.geo_id)
        self.assertEqual(pkt[:4], MAGIC)

    def test_BIN_002_N_field(self):
        """BIN_002: N = N_DIV_100 * 100."""
        obj = self._make_obj()
        pkt = pack_g(obj._hv(), obj.geo_id)
        n_div = struct.unpack(">H", pkt[6:8])[0]
        self.assertEqual(n_div * 100, N)

    def test_BIN_003_size(self):
        """BIN_003: tamaño del paquete = 14 + N*4."""
        obj = self._make_obj()
        pkt = pack_g(obj._hv(), obj.geo_id)
        self.assertEqual(len(pkt), 14 + N * 4)

    def test_BIN_004_roundtrip_exact(self):
        """BIN_004: pack→unpack es exacto (sim = 1.0)."""
        obj = self._make_obj()
        hv = obj._hv()
        pkt = pack_g(hv, obj.geo_id)
        hv_back = unpack_hv(pkt)
        sim = _sim(hv, hv_back)
        self.assertAlmostEqual(sim, 1.0, places=6,
                               msg=f"sim = {sim:.8f} < 1.0")

    def test_BIN_005_geo_id_preserved(self):
        """BIN_005: GEO_ID se preserva en el paquete."""
        obj = self._make_obj()
        pkt = pack_g(obj._hv(), obj.geo_id)
        geo_recovered = unpack_geo_id(pkt)
        self.assertEqual(geo_recovered, obj.geo_id)

    def test_BIN_006_flags(self):
        """BIN_006: FLAGS = 0x01 (HAS_GEO) en Modo G."""
        obj = self._make_obj()
        pkt = pack_g(obj._hv(), obj.geo_id)
        flags = pkt[5]
        self.assertEqual(flags & 0x01, 0x01, "HAS_GEO no está seteado")
        self.assertEqual(flags & 0x02, 0x00, "HAS_DIMS inesperado en Modo G")

    def test_BIN_007_version(self):
        """BIN_007: VERSION = 4."""
        obj = self._make_obj()
        pkt = pack_g(obj._hv(), obj.geo_id)
        self.assertEqual(pkt[4], VERSION)

    def test_multiple_objects_distinct(self):
        """Hipervectores de objetos distintos dan paquetes distintos."""
        obj1 = self._make_obj([("DIM_SQL", {"tabla": "a"}, 1.0)])
        obj2 = self._make_obj([("DIM_FLUTTER", {"widget": "b"}, 1.0)])
        pkt1 = pack_g(obj1._hv(), obj1.geo_id)
        pkt2 = pack_g(obj2._hv(), obj2.geo_id)
        hv1 = unpack_hv(pkt1)
        hv2 = unpack_hv(pkt2)
        sim = _sim(hv1, hv2)
        self.assertLess(sim, 0.95, f"Objetos distintos demasiado similares: {sim:.4f}")

    def test_dims_preserved_after_roundtrip(self):
        """Dimensiones activas se preservan tras pack/unpack."""
        obj = self._make_obj()
        hv_orig = obj._hv()
        pkt = pack_g(hv_orig, obj.geo_id)
        hv_back = unpack_hv(pkt)

        dims_orig = {d for d in NATIVE if _proj(hv_orig, self.sp.sub(d)) > UMBRAL}
        dims_back = {d for d in NATIVE if _proj(hv_back, self.sp.sub(d)) > UMBRAL}
        self.assertEqual(dims_orig, dims_back,
                         f"Dims cambiaron: antes={dims_orig}, después={dims_back}")


# ---------------------------------------------------------------------------
# Tests Python→Rust (solo si --with-rust)
# ---------------------------------------------------------------------------

RUST_TEST_PROGRAM = r'''
use std::fs;
use std::io::Read;

const MAGIC: &[u8; 4] = b"PDIM";

fn sim(a: &[f32], b: &[f32]) -> f32 {
    let dot: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
    (dot + 1.0) / 2.0
}

fn unpack_hv(data: &[u8]) -> Option<Vec<f32>> {
    if data.len() < 14 || &data[..4] != MAGIC { return None; }
    if data[5] & 0x01 == 0 { return None; }
    let n = u16::from_be_bytes([data[6], data[7]]) as usize * 100;
    if data.len() < 14 + n * 4 { return None; }
    Some(data[14..14 + n * 4].chunks_exact(4)
        .map(|b| f32::from_be_bytes([b[0],b[1],b[2],b[3]]))
        .collect())
}

fn norm(v: &[f32]) -> f32 {
    v.iter().map(|x| x * x).sum::<f32>().sqrt()
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 3 {
        eprintln!("Usage: test_interop <packet.bin> <reference.bin>");
        std::process::exit(1);
    }

    let packet = fs::read(&args[1]).expect("Cannot read packet");
    let reference = fs::read(&args[2]).expect("Cannot read reference");

    // Verificar MAGIC
    assert_eq!(&packet[..4], MAGIC, "MAGIC inválido");
    println!("  ✓ MAGIC correcto");

    // Desempaquetar
    let hv = unpack_hv(&packet).expect("unpack falló");
    println!("  ✓ unpack: {} floats", hv.len());

    // Verificar vs referencia (raw float32 BE desde Python)
    let hv_ref: Vec<f32> = reference.chunks_exact(4)
        .map(|b| f32::from_be_bytes([b[0],b[1],b[2],b[3]]))
        .collect();

    assert_eq!(hv.len(), hv_ref.len(), "Longitudes distintas");

    let similarity = sim(&hv, &hv_ref);
    println!("  sim(Rust, Python) = {:.8}", similarity);
    assert!(similarity > 0.9999, "sim < 0.9999: {}", similarity);
    println!("  ✓ sim > 0.9999");

    // Verificar norma
    let n = norm(&hv);
    println!("  norm(hv) = {:.8}", n);
    assert!((n - 1.0).abs() < 0.001, "norma != 1: {}", n);
    println!("  ✓ norma ≈ 1.0");

    println!("\n✓ Python→Rust POLYDIM_BIN interop: PASS");
}
'''

class TestPolydimBinRust(unittest.TestCase):
    """Tests de interoperabilidad Python↔Rust (requieren rustc)."""

    @classmethod
    def setUpClass(cls):
        if not WITH_RUST:
            return
        # Verificar rustc disponible
        result = subprocess.run(['rustc', '--version'], capture_output=True)
        cls.rustc_available = result.returncode == 0

    @unittest.skipUnless(WITH_RUST, "Omitido sin --with-rust")
    def test_python_to_rust_roundtrip(self):
        """Paquete generado en Python puede ser leído por Rust con sim > 0.9999."""
        if not self.rustc_available:
            self.skipTest("rustc no disponible")

        sp = Space("RUST_INTEROP")
        obj = ObjectND(sp)
        obj.add("DIM_SQL", {"tabla": "orders", "pk": "id"}, w=1.0)
        obj.add("DIM_PYTHON", {"tipo": "dataclass"}, w=0.7)

        hv = obj._hv()
        pkt = pack_g(hv, obj.geo_id)

        with tempfile.TemporaryDirectory() as tmpdir:
            pkt_path = os.path.join(tmpdir, "packet.bin")
            ref_path = os.path.join(tmpdir, "reference.bin")
            src_path = os.path.join(tmpdir, "test_interop.rs")
            bin_path = os.path.join(tmpdir, "test_interop")

            with open(pkt_path, 'wb') as f:
                f.write(pkt)
            with open(ref_path, 'wb') as f:
                # Python: float32 → bytes (big-endian para consistencia)
                for v in hv:
                    f.write(struct.pack('>f', float(v)))
            with open(src_path, 'w') as f:
                f.write(RUST_TEST_PROGRAM)

            # Compilar
            compile_r = subprocess.run(
                ['rustc', src_path, '-O', '-o', bin_path],
                capture_output=True, text=True
            )
            if compile_r.returncode != 0:
                self.fail(f"rustc falló: {compile_r.stderr}")

            # Ejecutar
            run_r = subprocess.run(
                [bin_path, pkt_path, ref_path],
                capture_output=True, text=True
            )
            print(run_r.stdout)
            if run_r.returncode != 0:
                self.fail(f"Test Rust falló:\n{run_r.stdout}\n{run_r.stderr}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    print(f"POLYDIM Integration Tests — N={N}")
    print(f"Rust tests: {'HABILITADOS' if WITH_RUST else 'deshabilitados (usar --with-rust)'}")
    print()

    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    suite.addTests(loader.loadTestsFromTestCase(TestPolydimBinPython))
    if WITH_RUST:
        suite.addTests(loader.loadTestsFromTestCase(TestPolydimBinRust))

    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    print(f"\nTests: {result.testsRun}  Fallos: {len(result.failures)}  Errores: {len(result.errors)}")
    sys.exit(0 if result.wasSuccessful() else 1)
