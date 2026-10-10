import random
import math

# =====================================
# 1. DATA LATIH ACAK
# =====================================
data = []

for i in range(20):
    if i < 10:
        x = random.uniform(0.5, 5.4)
        y = random.uniform(0.5, 5.5)
        kelas = "Belum Matang"
    else:
        x = random.uniform(5.2, 9.7)
        y = random.uniform(5.0, 9.7)
        kelas = "Matang"

    data.append({
        "id": chr(65 + i),
        "x": round(x, 2),
        "y": round(y, 2),
        "kelas": kelas
    })

random.shuffle(data)

for i, p in enumerate(data):
    p["id"] = chr(65 + i)

print("=== DATA LATIH ===")
for p in data:
    print(p)

# =====================================
# 2. INPUT BUAH UJI
# =====================================
qx = float(input("\nMasukkan X buah uji (0-10): "))
qy = float(input("Masukkan Y buah uji (0-10): "))
k = int(input("Masukkan nilai K (1, 3, atau 5): "))

qx = max(0, min(10, qx))
qy = max(0, min(10, qy))
k = max(1, min(20, k))

# =====================================
# 3. MENGHITUNG JARAK EUCLIDEAN
# =====================================
def jarak(x1, y1, x2, y2):
    return math.sqrt((x1-x2)**2 + (y1-y2)**2)

hasil = []

for p in data:
    d = jarak(qx, qy, p["x"], p["y"])
    hasil.append((d, p))

hasil.sort(key=lambda item: item[0])
tetangga = hasil[:k]

print("\n=== PERHITUNGAN JARAK ===")
for d, p in hasil:
    rumus = (
        f"√[({qx:.2f}-{p['x']:.2f})² + "
        f"({qy:.2f}-{p['y']:.2f})²]"
    )
    dipilih = "Ya" if (d, p) in tetangga else "Tidak"
    print(
        f"{p['id']} | ({p['x']:.2f},{p['y']:.2f}) | "
        f"{rumus} = {d:.3f} | {p['kelas']} | Dipilih: {dipilih}"
    )

# =====================================
# 4. KLASIFIKASI KNN
# =====================================
suara = {"Matang": 0, "Belum Matang": 0}

for d, p in tetangga:
    suara[p["kelas"]] += 1

if suara["Matang"] == suara["Belum Matang"]:
    prediksi = tetangga[0][1]["kelas"]
else:
    prediksi = max(suara, key=suara.get)

print("\n=== HASIL KNN ===")
print("Koordinat buah uji:", (qx, qy))
print("Nilai K:", k)
print("Suara kelas:", suara)
print("Prediksi:", prediksi)

# =====================================
# 5. MEMBANGUN KD-TREE
# =====================================
def buat_tree(points, depth=0):
    if not points:
        return None

    sumbu = "x" if depth % 2 == 0 else "y"
    points = sorted(points, key=lambda p: p[sumbu])
    tengah = len(points) // 2

    return {
        "point": points[tengah],
        "sumbu": sumbu,
        "kiri": buat_tree(points[:tengah], depth + 1),
        "kanan": buat_tree(points[tengah+1:], depth + 1)
    }

root = buat_tree(data)

# =====================================
# 6. PENCARIAN KD-TREE
# =====================================
dikunjungi = set()
kandidat = []

def cari_tree(node):
    if node is None:
        return

    p = node["point"]
    dikunjungi.add(p["id"])

    d = jarak(qx, qy, p["x"], p["y"])
    kandidat.append((d, p))
    kandidat.sort(key=lambda item: item[0])

    if len(kandidat) > k:
        kandidat.pop()

    sumbu = node["sumbu"]
    delta = (qx - p["x"]) if sumbu == "x" else (qy - p["y"])

    if delta < 0:
        dekat, jauh = node["kiri"], node["kanan"]
    else:
        dekat, jauh = node["kanan"], node["kiri"]

    cari_tree(dekat)

    batas = kandidat[-1][0] if len(kandidat) == k else float("inf")
    if abs(delta) < batas:
        cari_tree(jauh)

cari_tree(root)

print("\n=== HASIL PENCARIAN KD-TREE ===")
print("Node dikunjungi:", len(dikunjungi), "dari", len(data))

for d, p in kandidat:
    print(f"{p['id']} | Jarak: {d:.3f} | {p['kelas']}")

# =====================================
# 7. GRAFIK KOORDINAT TEKS
# =====================================
print("\n=== GRAFIK KOORDINAT X-Y ===")
print("M = Matang | B = Belum Matang | Q = Buah uji")

ukuran = 11
grid = [["." for _ in range(ukuran)] for _ in range(ukuran)]

for p in data:
    x = round(p["x"])
    y = round(p["y"])
    simbol = "M" if p["kelas"] == "Matang" else "B"
    grid[10-y][x] = simbol

grid[10-round(qy)][round(qx)] = "Q"

print(" Y/X  " + " ".join(str(i) for i in range(11)))
for i, baris in enumerate(grid):
    print(f" {10-i:2}   " + "  ".join(baris))

# =====================================
# 8. GRAFIK BATANG JARAK
# =====================================
print("\n=== GRAFIK JARAK TETANGGA ===")

maks = max((d for d, p in hasil), default=1)
for i, (d, p) in enumerate(hasil[:10]):
    panjang = int(d / maks * 30) if maks else 0
    tanda = "*" if i < k else "-"
    print(f"{p['id']} {tanda} | {'#' * panjang} {d:.3f}")

# =====================================
# 9. TAMPILAN STRUKTUR KD-TREE
# =====================================
print("\n=== STRUKTUR KD-TREE ===")

def tampil_tree(node, level=0):
    if node is None:
        return

    p = node["point"]
    tanda = "*" if p["id"] in dikunjungi else " "
    print(
        "  " * level + f"{tanda}{p['id']} "
        f"({p['x']:.1f},{p['y']:.1f}) "
        f"[{node['sumbu'].upper()}]"
    )
    tampil_tree(node["kiri"], level + 1)
    tampil_tree(node["kanan"], level + 1)

tampil_tree(root)