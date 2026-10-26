# =====================================================================
# KNN (K-NEAREST NEIGHBOR) + KD-TREE
# =====================================================================

# CARA MENUNJUK BAGIAN PENTING SAAT PRESENTASI:
#   tekan Ctrl+F lalu cari  "### KUNCI"  -> semua bagian inti muncul.

#   KUNCI 1   data latih
#   KUNCI 2   input titik uji Q dan nilai K
#   KUNCI 3   KNN BRUTE FORCE    : hitung SEMUA jarak -> urutkan -> ambil K
#   KUNCI 4   VOTING             : kelas mayoritas = prediksi
#   KUNCI 5   MEMBANGUN KD-TREE  : bagi di median, sumbu X/Y bergantian
#   KUNCI 6   MENCARI DI KD-TREE : telusuri sisi dekat, PANGKAS sisi jauh
#   KUNCI 7-9 tampilan hasil (grafik teks dan struktur pohon)
#   KUNCI 10  3 kasus uji kecil yang bisa dicek manual
#   KUNCI 11  galat klasifikasi dan laju konvergensi
#   KUNCI 12  PERBANDINGAN DENGAN DECISION TREE C4.5 (galat, waktu, aturan if-then)

# ISTILAH SINGKAT:
#   Q / titik uji : buah baru yang kelasnya mau ditebak (qx, qy)
#   tetangga      : titik latih yang paling dekat dengan Q
#   K             : berapa banyak tetangga yang diminta pendapatnya
#   jarak         : jarak Euclidean = akar((qx-px)^2 + (qy-py)^2)
#   brute force   : periksa SEMUA titik, tanpa trik
#   KD-Tree       : pohon yang membagi ruang supaya tidak semua titik diperiksa
#   pruning       : memangkas (melewati) cabang yang pasti tidak berisi tetangga
#   median        : titik tengah setelah data diurutkan
#   rekursi       : fungsi yang memanggil dirinya sendiri
#   entropi       : ukuran "campur aduk" kelas dalam satu kelompok data (0 = murni)
#   gain ratio    : kriteria C4.5 untuk memilih pemisah (gain dibagi split info)
#   aturan if-then: bentuk pohon keputusan yang bisa dibaca manusia
# =====================================================================

import random
import math

# =====================================
# 1. DATA LATIH ACAK
# =====================================
# ### KUNCI 1 ### DATA LATIH
# Data latih = buah contoh yang statusnya SUDAH diketahui. Tiap titik disimpan
# sebagai dictionary: id, x, y, kelas. Semua disimpan di daftar "data".
data = []

# Membuat 20 titik: 10 pertama "Belum Matang", 10 berikutnya "Matang".
for i in range(20):
    if i < 10:
        # Belum Matang: X dan Y acak di sudut kiri-bawah (nilai kecil).
        # random.uniform(a, b) = bilangan desimal acak antara a dan b.
        x = random.uniform(0.5, 5.4)
        y = random.uniform(0.5, 5.5)
        kelas = "Belum Matang"
    else:
        # Matang: X dan Y acak di sudut kanan-atas (nilai besar).
        # Rentangnya SEDIKIT tumpang tindih dengan Belum Matang (X 5,2-5,4 dan
        # Y 5,0-5,5) = daerah "setengah matang" yang membingungkan.
        x = random.uniform(5.2, 9.7)
        y = random.uniform(5.0, 9.7)
        kelas = "Matang"

    # round(x, 2) = dibulatkan 2 desimal. chr(65 + i) = huruf: 65 -> A, 66 -> B, ...
    data.append({
        "id": chr(65 + i),
        "x": round(x, 2),
        "y": round(y, 2),
        "kelas": kelas
    })

# Urutan diacak agar posisi dalam daftar tidak membocorkan kelasnya.
random.shuffle(data)

# ID dibagi ulang (A, B, C...) sesudah diacak supaya urut mengikuti daftar baru.
for i, p in enumerate(data):
    p["id"] = chr(65 + i)

print("=== DATA LATIH ===")
for p in data:
    print(p)

# =====================================
# 2. INPUT BUAH UJI
# =====================================
# ### KUNCI 2 ### INPUT TITIK UJI Q DAN NILAI K
# Q = buah baru yang mau ditebak. input() membaca ketikan; float = desimal.
qx = float(input("\nMasukkan X buah uji (0-10): "))
qy = float(input("Masukkan Y buah uji (0-10): "))
# K = jumlah tetangga terdekat yang diminta pendapatnya (bilangan bulat).
# Contoh: K=3 -> tebak kelas dari 3 titik latih terdekat.
# Untuk 2 kelas, pilih K GANJIL agar suara tidak seri.
k = int(input("Masukkan nilai K (1, 3, atau 5): "))

# Pengaman: memaksa nilai tetap di 0..10 (min memotong atas, max memotong bawah).
qx = max(0, min(10, qx))
qy = max(0, min(10, qy))
# K minimal 1 dan maksimal 20 (tidak boleh melebihi jumlah data latih).
k = max(1, min(20, k))

# =====================================
# 3. MENGHITUNG JARAK EUCLIDEAN
# =====================================
# ### KUNCI 3 ### KNN BRUTE FORCE (mulai dari fungsi jarak ini)
# Jarak Euclidean = garis lurus antara dua titik (rumus Pythagoras).
# Makin kecil jarak, makin MIRIP kedua buah.
def jarak(x1, y1, x2, y2):
    return math.sqrt((x1-x2)**2 + (y1-y2)**2)

# --- BRUTE FORCE: periksa SEMUA titik, satu per satu, tanpa ada yang dilewati ---
# hasil akan berisi pasangan (jarak, titik) untuk SEMUA 20 titik latih.
hasil = []

# Loop ini menyentuh setiap titik latih -> inilah arti "brute force".
# Jumlah jarak yang dihitung = n (di sini 20).
for p in data:
    d = jarak(qx, qy, p["x"], p["y"])
    hasil.append((d, p))

# Urutkan dari jarak TERKECIL. item[0] = jarak (elemen pertama pasangan).
# Pengurutan inilah yang membuat brute force berbiaya O(n log n).
hasil.sort(key=lambda item: item[0])
# >>> INI K TETANGGA TERDEKAT: ambil K data teratas dari daftar terurut. <<<
# hasil[:k] = potong daftar dari awal sampai K elemen.
tetangga = hasil[:k]

# Mencetak tabel: rumus + hasil jarak + kelas + apakah dipilih (untuk dicek manual).
print("\n=== PERHITUNGAN JARAK ===")
for d, p in hasil:
    rumus = (
        f"√[({qx:.2f}-{p['x']:.2f})² + "
        f"({qy:.2f}-{p['y']:.2f})²]"
    )
    # "Ya" jika pasangan (jarak, titik) ini termasuk K tetangga terpilih.
    dipilih = "Ya" if (d, p) in tetangga else "Tidak"
    print(
        f"{p['id']} | ({p['x']:.2f},{p['y']:.2f}) | "
        f"{rumus} = {d:.3f} | {p['kelas']} | Dipilih: {dipilih}"
    )

# =====================================
# 4. KLASIFIKASI KNN
# =====================================
# ### KUNCI 4 ### VOTING (KLASIFIKASI)
# Dua penghitung suara, awalnya 0. Tiap tetangga memberi 1 suara untuk kelasnya.
suara = {"Matang": 0, "Belum Matang": 0}

# Hanya K tetangga yang boleh memilih, bukan semua titik.
for d, p in tetangga:
    suara[p["kelas"]] += 1

# Jika suara SERI (hanya mungkin saat K genap): ikuti kelas tetangga TERDEKAT
# (tetangga[0] = jarak terkecil). Kasus ini diuji di Kasus Uji 3.
if suara["Matang"] == suara["Belum Matang"]:
    prediksi = tetangga[0][1]["kelas"]
else:
    # Jika tidak seri: pilih kelas dengan suara terbanyak (mayoritas).
    # >>> "prediksi" di program ini berasal dari BRUTE FORCE (bukan dari KD-Tree). <<<
    prediksi = max(suara, key=suara.get)

# Mencetak hasil akhir: koordinat Q, K, perolehan suara, dan prediksi.
print("\n=== HASIL KNN ===")
print("Koordinat buah uji:", (qx, qy))
print("Nilai K:", k)
print("Suara kelas:", suara)
print("Prediksi:", prediksi)

# =====================================
# 5. MEMBANGUN KD-TREE
# =====================================
# ### KUNCI 5 ### MEMBANGUN KD-TREE
# Tujuan: menyusun titik latih menjadi POHON supaya pencarian tidak perlu
# memeriksa semua titik. depth = kedalaman (0 = akar). Fungsi ini REKURSIF.
def buat_tree(points, depth=0):
    # Syarat berhenti: kalau tidak ada titik tersisa, tidak ada simpul (None).
    if not points:
        return None

    # Sumbu BERGANTIAN: kedalaman genap bagi menurut X, ganjil menurut Y.
    # Supaya kedua dimensi ikut membagi ruang.
    sumbu = "x" if depth % 2 == 0 else "y"
    # Urutkan titik menurut sumbu yang dipakai.
    points = sorted(points, key=lambda p: p[sumbu])
    # Indeks MEDIAN (titik tengah). "//" = pembagian bulat.
    # Membagi di median membuat pohon SEIMBANG (tinggi sekitar log2 n).
    tengah = len(points) // 2

    # Satu SIMPUL pohon berisi: titik median, sumbu pembaginya, serta anak kiri dan kanan.
    return {
        "point": points[tengah],
        "sumbu": sumbu,
        # Anak kiri = pohon dari titik-titik SEBELUM median (nilai sumbu lebih kecil).
        # Memanggil dirinya sendiri dengan depth+1 (inilah rekursi).
        "kiri": buat_tree(points[:tengah], depth + 1),
        # Anak kanan = pohon dari titik-titik SESUDAH median (nilai sumbu lebih besar).
        "kanan": buat_tree(points[tengah+1:], depth + 1)
    }

# Memanggil fungsi untuk seluruh data; hasilnya akar (root) pohon.
root = buat_tree(data)

# =====================================
# 6. PENCARIAN KD-TREE
# =====================================
# ### KUNCI 6 ### MENCARI K TETANGGA DI KD-TREE (+ PEMANGKASAN)
# dikunjungi = catatan simpul yang benar-benar dihitung jaraknya.
# Dibandingkan dengan n, ini membuktikan berapa banyak yang berhasil DILEWATI.
dikunjungi = set()
# kandidat = daftar K tetangga TERBAIK SEJAUH INI (terurut). Awalnya kosong.
kandidat = []

# Rekursif: menelusuri pohon mulai dari akar, memutuskan cabang mana dikunjungi.
def cari_tree(node):
    # Syarat berhenti: cabang kosong -> tidak ada yang dikerjakan.
    if node is None:
        return

    # Titik pada simpul ini. Catat bahwa simpul ini dikunjungi.
    p = node["point"]
    dikunjungi.add(p["id"])

    # Hitung jarak Q ke titik simpul ini, masukkan ke kandidat, lalu urutkan.
    d = jarak(qx, qy, p["x"], p["y"])
    kandidat.append((d, p))
    kandidat.sort(key=lambda item: item[0])

    # Jika kandidat lebih dari K, buang yang TERJAUH (pop mengambil elemen terakhir).
    # Jadi kandidat selalu berisi paling banyak K titik terbaik.
    if len(kandidat) > k:
        kandidat.pop()

    # Sumbu pembagi simpul ini (x atau y).
    sumbu = node["sumbu"]
    # delta = selisih koordinat Q dengan GARIS PEMBAGI pada sumbu simpul.
    # Tandanya menunjukkan Q berada di sisi mana; besarnya |delta| = jarak Q ke garis.
    delta = (qx - p["x"]) if sumbu == "x" else (qy - p["y"])

    # Menentukan sisi DEKAT (sisi tempat Q berada) dan sisi JAUH (seberang garis).
    # delta negatif -> Q di kiri -> kiri = dekat, kanan = jauh. Positif -> sebaliknya.
    if delta < 0:
        dekat, jauh = node["kiri"], node["kanan"]
    else:
        dekat, jauh = node["kanan"], node["kiri"]

    # SELALU telusuri sisi DEKAT dulu: tetangga terdekat paling mungkin ada di sana.
    cari_tree(dekat)

    # batas = jarak kandidat TERBURUK saat ini (kandidat[-1] = yang terjauh).
    # Selama kandidat belum berisi K titik, batas = tak hingga (jangan memangkas dulu).
    batas = kandidat[-1][0] if len(kandidat) == k else float("inf")
    # >>> ### KUNCI 6 ### PRUNING (PEMANGKASAN) -- BAGIAN TERPENTING KD-TREE <<<
    # Semua titik di sisi SEBERANG garis pembagi berjarak paling sedikit |delta| dari Q.
    # Jika |delta| >= batas, tidak mungkin ada titik di sana yang lebih dekat dari
    # kandidat terburuk -> sisi jauh DILEWATI tanpa menghitung satu jaraknya pun.
    # Jika |delta| < batas, sisi jauh MASIH mungkin berisi tetangga -> dikunjungi.
    # Contoh nyata (Kasus Uji 1): di titik A, delta = 3,0 >= batas 2,55 -> cabang B dipangkas.
    if abs(delta) < batas:
        cari_tree(jauh)

# Menjalankan pencarian mulai dari akar.
cari_tree(root)

# Menampilkan hasil KD-Tree. Seharusnya SAMA dengan hasil brute force (K tetangga yang sama),
# tetapi jumlah simpul dikunjungi lebih sedikit dari n -> itulah penghematannya.
print("\n=== HASIL PENCARIAN KD-TREE ===")
print("Node dikunjungi:", len(dikunjungi), "dari", len(data))

for d, p in kandidat:
    print(f"{p['id']} | Jarak: {d:.3f} | {p['kelas']}")

# =====================================
# 7. GRAFIK KOORDINAT TEKS
# =====================================
# ### KUNCI 7 ### GRAFIK KOORDINAT TEKS (hanya tampilan, bukan bagian algoritma)
# M = Matang, B = Belum Matang, Q = titik uji. Koordinat dibulatkan, jadi dua titik
# bisa menempati sel yang sama.
print("\n=== GRAFIK KOORDINAT X-Y ===")
print("M = Matang | B = Belum Matang | Q = Buah uji")

# Grid 11 x 11 untuk koordinat 0..10, awalnya berisi titik "."
ukuran = 11
grid = [["." for _ in range(ukuran)] for _ in range(ukuran)]

for p in data:
    x = round(p["x"])
    y = round(p["y"])
    simbol = "M" if p["kelas"] == "Matang" else "B"
    grid[10-y][x] = simbol

# Tandai posisi titik uji dengan Q (baris dibalik: Y besar di atas).
grid[10-round(qy)][round(qx)] = "Q"

print(" Y/X  " + " ".join(str(i) for i in range(11)))
for i, baris in enumerate(grid):
    print(f" {10-i:2}   " + "  ".join(baris))

# =====================================
# 8. GRAFIK BATANG JARAK
# =====================================
# ### KUNCI 8 ### GRAFIK BATANG JARAK (tampilan). Panjang "#" sebanding dengan jarak.
# Tanda "*" = termasuk K tetangga terpilih, "-" = tidak.
print("\n=== GRAFIK JARAK TETANGGA ===")

# Jarak terjauh dipakai sebagai patokan panjang batang (30 karakter).
maks = max((d for d, p in hasil), default=1)
for i, (d, p) in enumerate(hasil[:10]):
    panjang = int(d / maks * 30) if maks else 0
    tanda = "*" if i < k else "-"
    print(f"{p['id']} {tanda} | {'#' * panjang} {d:.3f}")

# =====================================
# 9. TAMPILAN STRUKTUR KD-TREE
# =====================================
# ### KUNCI 9 ### MENCETAK STRUKTUR POHON (tampilan)
# Indentasi = kedalaman, [X]/[Y] = sumbu pembagi, tanda "*" = simpul yang dikunjungi saat pencarian.
print("\n=== STRUKTUR KD-TREE ===")

# Rekursif: cetak simpul, lalu anak kiri, lalu anak kanan.
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

# Simpul tanpa "*" = berhasil DIPANGKAS (tidak dihitung jaraknya).
tampil_tree(root)


# =====================================
# 10. KASUS UJI KECIL (BISA DICEK MANUAL)
# =====================================
# Dataset tetap 8 titik (bukan acak) agar semua angka dapat
# dihitung dengan kalkulator dan dicocokkan dengan keluaran program.
# ### KUNCI 10 ### KASUS UJI KECIL (BISA DICEK MANUAL)
# Tujuan: membuktikan program benar. Dataset TETAP 8 titik (bukan acak), jadi semua
# angka bisa dihitung dengan kalkulator lalu dibandingkan dengan keluaran program.
data_uji = [
    {"id": "A", "x": 2.0, "y": 3.0, "kelas": "Belum Matang"},
    {"id": "B", "x": 1.5, "y": 1.0, "kelas": "Belum Matang"},
    {"id": "C", "x": 4.0, "y": 4.5, "kelas": "Belum Matang"},
    {"id": "D", "x": 3.0, "y": 5.0, "kelas": "Belum Matang"},
    {"id": "E", "x": 6.0, "y": 6.5, "kelas": "Matang"},
    {"id": "F", "x": 7.5, "y": 8.0, "kelas": "Matang"},
    {"id": "G", "x": 8.0, "y": 5.5, "kelas": "Matang"},
    {"id": "H", "x": 5.5, "y": 7.5, "kelas": "Matang"},
]

# Nilai "diharapkan" di bawah ini adalah HASIL HITUNG TANGAN.
# Tiap kasus menyimpan nilai YANG DIHARAPKAN hasil hitung tangan:
# tetangga, jarak, suara, prediksi, dan jumlah simpul dikunjungi.
# Kasus 1 = dekat Belum Matang | Kasus 2 = dekat Matang | Kasus 3 = suara seri (K genap).
kasus_uji = [
    {
        "nama": "Kasus 1: dekat Belum Matang",
        "q": (5.0, 5.0), "k": 3,
        "tetangga": ["C", "E", "D"],
        "jarak": [1.118, 1.803, 2.000],
        "suara": {"Matang": 1, "Belum Matang": 2},
        "prediksi": "Belum Matang",
        "dikunjungi": 7,            # hanya B yang dipangkas
    },
    {
        "nama": "Kasus 2: dekat Matang (suara bulat)",
        "q": (6.5, 7.0), "k": 3,
        "tetangga": ["E", "H", "F"],
        "jarak": [0.707, 1.118, 1.414],
        "suara": {"Matang": 3, "Belum Matang": 0},
        "prediksi": "Matang",
        "dikunjungi": 6,            # cabang A dan B dipangkas
    },
    {
        "nama": "Kasus 3: seri suara (K genap)",
        "q": (5.0, 5.8), "k": 2,
        "tetangga": ["E", "C"],
        "jarak": [1.221, 1.640],
        "suara": {"Matang": 1, "Belum Matang": 1},
        "prediksi": "Matang",      # seri -> ikut tetangga terdekat (E)
        "dikunjungi": 7,            # hanya B yang dipangkas
    },
]

# Menjalankan ketiga kasus: brute force, KD-Tree, lalu membandingkan dengan nilai manual.
def jalankan_kasus_uji():
    # "global" karena cari_tree memakai variabel global. Nilai asli disimpan dulu lalu
    # dipulihkan di akhir agar program utama tidak terganggu.
    global qx, qy, k, kandidat, dikunjungi
    asli = (qx, qy, k, kandidat, dikunjungi)   # simpan nilai program utama
    pohon_uji = buat_tree(data_uji)
    lulus_semua = True

    print("\n" + "=" * 50)
    print("=== KASUS UJI KECIL (DATASET TETAP 8 TITIK) ===")
    print("=" * 50)

    for kasus in kasus_uji:
        qx, qy = kasus["q"]
        k = kasus["k"]
        print(f"\n{kasus['nama']}  |  Q=({qx},{qy})  K={k}")

        # Hitung semua jarak, urutkan, ambil K, lalu voting (sama dengan KUNCI 3 dan 4).
        # --- brute force ---
        hasil_uji = sorted(
            ((jarak(qx, qy, p["x"], p["y"]), p) for p in data_uji),
            key=lambda item: item[0]
        )
        for i, (d, p) in enumerate(hasil_uji):
            rumus = (f"√[({qx:.1f}-{p['x']:.1f})² + "
                     f"({qy:.1f}-{p['y']:.1f})²]")
            tanda = "Ya" if i < k else "Tidak"
            print(f"  {p['id']} | {rumus} = {d:.3f} | {p['kelas']} | Dipilih: {tanda}")

        tetangga_uji = hasil_uji[:k]
        suara_uji = {"Matang": 0, "Belum Matang": 0}
        for d, p in tetangga_uji:
            suara_uji[p["kelas"]] += 1
        if suara_uji["Matang"] == suara_uji["Belum Matang"]:
            prediksi_uji = tetangga_uji[0][1]["kelas"]
        else:
            prediksi_uji = max(suara_uji, key=suara_uji.get)

        # Reset kandidat dan dikunjungi, lalu cari lewat KD-Tree (sama dengan KUNCI 6).
        # --- KD-Tree ---
        kandidat = []
        dikunjungi = set()
        cari_tree(pohon_uji)

        # Enam pemeriksaan per kasus; tiap hasil dicetak LULUS atau GAGAL.
        # --- pemeriksaan terhadap hitungan manual ---
        pemeriksaan = [
            ("Tetangga brute force",
             [p["id"] for d, p in tetangga_uji] == kasus["tetangga"]),
            ("Jarak tetangga (3 desimal)",
             [round(d, 3) for d, p in tetangga_uji] == kasus["jarak"]),
            ("Suara",
             suara_uji == kasus["suara"]),
            ("Prediksi",
             prediksi_uji == kasus["prediksi"]),
            ("KD-Tree = brute force",
             [p["id"] for d, p in kandidat] == kasus["tetangga"]),
            ("Simpul dikunjungi",
             len(dikunjungi) == kasus["dikunjungi"]),
        ]
        print(f"  Suara: {suara_uji} -> Prediksi: {prediksi_uji}")
        print(f"  KD-Tree: dikunjungi {len(dikunjungi)} dari {len(data_uji)}, "
              f"hasil {[p['id'] for d, p in kandidat]}")
        for nama, ok in pemeriksaan:
            print(f"    [{'LULUS' if ok else 'GAGAL'}] {nama}")
            lulus_semua = lulus_semua and ok

    qx, qy, k, kandidat, dikunjungi = asli     # pulihkan nilai program utama
    print("\nRINGKASAN:", "SEMUA KASUS UJI LULUS" if lulus_semua
          else "ADA KASUS UJI YANG GAGAL")

# Dijalankan otomatis setiap program selesai.
jalankan_kasus_uji()


# =====================================
# 11. GALAT KLASIFIKASI DAN LAJU KONVERGENSI
# =====================================
# Galat klasifikasi = 100% - akurasi, diukur pada banyak titik uji
# acak yang kelas aslinya diketahui. Pembangkit acak terpisah
# (rng) dengan seed tetap dipakai agar hasil dapat diulang dan
# tidak mengganggu data acak program utama.
# ### KUNCI 11 ### GALAT KLASIFIKASI DAN LAJU KONVERGENSI
# Galat klasifikasi = 100% - akurasi = persentase tebakan yang SALAH pada banyak titik uji
# yang kelas aslinya diketahui. rng = pembangkit acak terpisah (seed 42) agar hasil dapat
# diulang dan tidak mengganggu data acak program utama.
def bangkitkan_data(n, rng):
    hasil_data = []
    for i in range(n):
        if i < n // 2:
            hasil_data.append((rng.uniform(0.5, 5.4), rng.uniform(0.5, 5.5), "Belum Matang"))
        else:
            hasil_data.append((rng.uniform(5.2, 9.7), rng.uniform(5.0, 9.7), "Matang"))
    return hasil_data

# KNN versi ringkas (brute force) untuk eksperimen. Memakai d^2 tanpa akar karena
# urutan jarak sama; akar kuadrat tidak mengubah mana yang terdekat.
def prediksi_knn(data_latih, x, y, kk):
    # brute force; d^2 cukup karena akar kuadrat monoton naik
    urut = sorted(((x - a) ** 2 + (y - b) ** 2, c) for a, b, c in data_latih)
    tetangga_k = urut[:kk]
    sm = sum(1 for d2, c in tetangga_k if c == "Matang")
    sb = kk - sm
    if sm == sb:
        return tetangga_k[0][1]
    return "Matang" if sm > sb else "Belum Matang"

# Mengulang percobaan: buat data latih baru, uji banyak titik, hitung persen yang salah,
# lalu ambil rata-rata dan simpangan baku (seberapa bervariasi antar percobaan).
def galat_klasifikasi(n, kk, percobaan, jumlah_uji, rng):
    galat_list = []
    for _ in range(percobaan):
        latih = bangkitkan_data(n, rng)
        salah = 0
        for j in range(jumlah_uji):
            if j % 2 == 0:
                x, y, asli_kelas = rng.uniform(0.5, 5.4), rng.uniform(0.5, 5.5), "Belum Matang"
            else:
                x, y, asli_kelas = rng.uniform(5.2, 9.7), rng.uniform(5.0, 9.7), "Matang"
            if prediksi_knn(latih, x, y, kk) != asli_kelas:
                salah += 1
        galat_list.append(100.0 * salah / jumlah_uji)
    rata = sum(galat_list) / len(galat_list)
    sd = math.sqrt(sum((g - rata) ** 2 for g in galat_list) / (len(galat_list) - 1))
    return rata, sd

# [A] galat terhadap K (n=20): K mana yang paling sedikit salah.
# [B] galat terhadap n (K=3): seberapa cepat galat turun saat data latih bertambah.
def analisis_galat():
    rng = random.Random(42)
    print("\n" + "=" * 50)
    print("=== GALAT KLASIFIKASI (seed 42) ===")
    print("=" * 50)

    print("\n[A] Galat terhadap K  (n=20, 100 percobaan x 400 uji)")
    print(" K | galat rata-rata (%) | simpangan baku (%)")
    for kk in (1, 3, 5, 7):
        rata, sd = galat_klasifikasi(20, kk, 100, 400, rng)
        print(f" {kk} | {rata:19.2f} | {sd:18.2f} | {'#' * round(rata * 10)}")

    print("\n[B] Galat terhadap n  (K=3, 60 percobaan x 400 uji)")
    print("   n | galat rata-rata (%) | simpangan baku (%) | rasio ke n sebelumnya")
    daftar_n, daftar_galat = [], []
    sebelumnya = None
    for n in (6, 10, 20, 40, 80, 160, 320):
        rata, sd = galat_klasifikasi(n, 3, 60, 400, rng)
        rasio = f"{rata / sebelumnya:.2f}" if sebelumnya else "-"
        print(f" {n:3d} | {rata:19.2f} | {sd:18.2f} | {rasio:>21} | {'#' * round(rata * 10)}")
        daftar_n.append(n)
        daftar_galat.append(rata)
        sebelumnya = rata

    # LAJU KONVERGENSI: bila galat ~ c * n^(-p), maka log(galat) = log c - p * log(n) adalah garis
    # lurus; kemiringannya p dicari dengan regresi linear. p makin besar = galat turun makin cepat.
    # Galat tidak bisa turun di bawah "galat Bayes" (akibat daerah tumpang tindih kedua kelas).
    # laju konvergensi empiris: galat ~ c * n^(-p)  -> regresi log-log
    lx = [math.log(n) for n in daftar_n]
    ly = [math.log(g) for g in daftar_galat]
    mx, my = sum(lx) / len(lx), sum(ly) / len(ly)
    p = -sum((a - mx) * (b - my) for a, b in zip(lx, ly)) / sum((a - mx) ** 2 for a in lx)
    print(f"\nLaju konvergensi empiris: galat ~ n^(-{p:.2f})")
    print("Galat menurun seiring n bertambah dan menuju lantai kecil yang")
    print("berasal dari daerah tumpang tindih kedua kelas (galat Bayes).")

# Dijalankan otomatis. Hasil: galat ~ n^(-0,48), menuju sekitar 0,2 %.
analisis_galat()


# =====================================
# 12. PERBANDINGAN DENGAN DECISION TREE (C4.5)
# =====================================
# Decision Tree C4.5 ditulis dari nol (Python murni, tanpa pustaka).
# Ciri C4.5 yang dipakai:
#   - atribut dipilih dengan GAIN RATIO (gain dinormalkan dengan split info),
#     hanya dari atribut yang gain-nya >= rata-rata gain semua atribut;
#   - atribut numerik dibelah biner: x <= ambang  /  x > ambang, ambang dicari
#     di titik tengah dua nilai berurutan yang gain-nya terbesar;
#   - tiap cabang minimal 2 data (setara parameter -m 2 pada C4.5 asli).
# Penyederhanaan: tanpa pruning pessimistic, tanpa nilai hilang, tanpa atribut
# kategorikal (data kita hanya dua fitur numerik).
# Data memakai format yang sama dengan Bagian 11: tuple (x, y, kelas).
# ### KUNCI 12 ### PERBANDINGAN KNN vs DECISION TREE C4.5
# Decision Tree C4.5 = pohon KEPUTUSAN yang MEMBELAH data dengan pertanyaan ya/tidak
# (contoh: "x <= 5,35?") sampai kelompok data menjadi murni. Beda dengan KNN:
# KNN menghafal data lalu mencari tetangga (lazy), C4.5 merangkum data menjadi aturan (eager).
import time

# Entropi = seberapa CAMPUR kelas dalam satu kelompok. 0 = semua satu kelas (murni).
# Makin campur, makin besar nilainya (maksimum 1 untuk dua kelas sama banyak).
def entropi(hitung):
    total = sum(hitung.values())
    e = 0.0
    for c in hitung.values():
        if c:
            pr = c / total
            e -= pr * math.log2(pr)
    return e

# Mencari ambang pemisah TERBAIK untuk satu fitur (x atau y).
# Gain = entropi sebelum dibelah - entropi sesudah dibelah (seberapa banyak "campur" berkurang).
# Gain ratio = gain / split info; split info mencegah C4.5 menyukai pemisah yang memecah
# data menjadi bagian-bagian timpang. Ambang = titik tengah dua nilai berurutan.
def split_terbaik(data_latih, f, min_cabang):
    # kembalikan (gain, gain_ratio, ambang) terbaik untuk fitur f, atau None
    urut = sorted(data_latih, key=lambda r: r[f])
    n = len(urut)
    total = {}
    for r in urut:
        total[r[2]] = total.get(r[2], 0) + 1
    e_induk = entropi(total)
    kiri = {}
    terbaik = None
    for i in range(n - 1):
        kiri[urut[i][2]] = kiri.get(urut[i][2], 0) + 1
        if urut[i][f] == urut[i + 1][f]:
            continue
        nk, nr = i + 1, n - i - 1
        if nk < min_cabang or nr < min_cabang:
            continue
        kanan = {c: total[c] - kiri.get(c, 0) for c in total}
        gain = e_induk - (nk / n * entropi(kiri) + nr / n * entropi(kanan))
        split_info = -(nk / n * math.log2(nk / n) + nr / n * math.log2(nr / n))
        rasio = gain / split_info if split_info > 0 else 0.0
        if terbaik is None or gain > terbaik[0]:
            terbaik = (gain, rasio, (urut[i][f] + urut[i + 1][f]) / 2)
    return terbaik

# Membangun pohon C4.5 secara REKURSIF (sama pola dengan buat_tree KD-Tree, tetapi
# yang dipilih adalah pemisah dengan gain ratio terbaik, BUKAN median).
def bangun_c45(data_latih, min_cabang=2):
    hitung = {}
    for r in data_latih:
        hitung[r[2]] = hitung.get(r[2], 0) + 1
    mayoritas = max(hitung, key=hitung.get)
    if len(hitung) == 1 or len(data_latih) < 2 * min_cabang:
        return {"kelas": mayoritas}                       # daun
    calon = []
    for f in (0, 1):
        s = split_terbaik(data_latih, f, min_cabang)
        if s and s[0] > 1e-12:
            calon.append((f,) + s)                        # (fitur, gain, rasio, ambang)
    if not calon:
        return {"kelas": mayoritas}
    # Aturan khas C4.5: hanya atribut yang gain-nya >= rata-rata gain yang boleh bersaing,
    # lalu dipilih yang gain ratio-nya terbesar (di bawah).
    rata_gain = sum(c[1] for c in calon) / len(calon)
    layak = [c for c in calon if c[1] >= rata_gain - 1e-12]
    # Pemisah terpilih: fitur f dan ambang. Data dibagi menjadi kiri (<= ambang) dan kanan (> ambang).
    f, gain, rasio, ambang = max(layak, key=lambda c: c[2])   # gain ratio terbesar
    kiri = [r for r in data_latih if r[f] <= ambang]
    kanan = [r for r in data_latih if r[f] > ambang]
    return {"fitur": f, "ambang": ambang, "kelas": mayoritas,
            "kiri": bangun_c45(kiri, min_cabang),
            "kanan": bangun_c45(kanan, min_cabang)}

# PREDIKSI C4.5 = telusuri pohon dari akar mengikuti jawaban ya/tidak sampai daun.
# Hanya beberapa perbandingan (sedalam pohon) -> sangat cepat, tidak tergantung n.
def prediksi_c45(node, x, y):
    while "fitur" in node:
        nilai = x if node["fitur"] == 0 else y
        node = node["kiri"] if nilai <= node["ambang"] else node["kanan"]
    return node["kelas"]

def ukuran_c45(node):
    # (jumlah simpul, kedalaman)
    if "fitur" not in node:
        return 1, 0
    sk, dk = ukuran_c45(node["kiri"])
    sr, dr = ukuran_c45(node["kanan"])
    return 1 + sk + sr, 1 + max(dk, dr)

# Mencetak pohon sebagai aturan JIKA-MAKA. Inilah kelebihan C4.5: alasan keputusan terbaca.
def cetak_aturan(node, indent=0):
    pad = "   " * indent
    if "fitur" not in node:
        print(f"{pad}MAKA {node['kelas']}")
        return
    nama = "x" if node["fitur"] == 0 else "y"
    print(f"{pad}JIKA {nama} <= {node['ambang']:.2f}:")
    cetak_aturan(node["kiri"], indent + 1)
    print(f"{pad}SELAIN ITU ({nama} > {node['ambang']:.2f}):")
    cetak_aturan(node["kanan"], indent + 1)

def titik_uji_acak(j, rng):
    if j % 2 == 0:
        return rng.uniform(0.5, 5.4), rng.uniform(0.5, 5.5), "Belum Matang"
    return rng.uniform(5.2, 9.7), rng.uniform(5.0, 9.7), "Matang"

# KNN lewat KD-Tree (memakai buat_tree dan cari_tree dari KUNCI 5 dan 6) untuk pembandingan waktu.
def prediksi_kd(pohon, x, y, kk):
    # KNN lewat KD-Tree (memakai buat_tree dan cari_tree dari Bagian 5-6)
    global qx, qy, k, kandidat, dikunjungi
    qx, qy, k = x, y, kk
    kandidat = []
    dikunjungi = set()
    cari_tree(pohon)
    sm = sum(1 for d, p in kandidat if p["kelas"] == "Matang")
    sb = kk - sm
    if sm == sb:
        return kandidat[0][1]["kelas"]
    return "Matang" if sm > sb else "Belum Matang"

# Menjalankan tiga perbandingan: [A] galat, [B] waktu, [C] aturan if-then.
# Pembangkit acak terpisah (seed 42), nilai program utama disimpan lalu dipulihkan.
def perbandingan_c45():
    global qx, qy, k, kandidat, dikunjungi
    asli = (qx, qy, k, kandidat, dikunjungi)       # simpan nilai program utama
    rng = random.Random(42)
    print("\n" + "=" * 50)
    print("=== PERBANDINGAN KNN vs DECISION TREE C4.5 (seed 42) ===")
    print("=" * 50)

    # KNN dan C4.5 dilatih dan diuji pada data YANG SAMA sehingga perbandingannya adil.
    # Kolom "selisih" = galat KNN - galat C4.5; negatif berarti KNN lebih sedikit salah.
    # "+- " = galat baku selisih (makin kecil, makin meyakinkan).
    # [A] galat pada data latih dan titik uji YANG SAMA (berpasangan)
    print("\n[A] Galat klasifikasi (%), K=3, 60 percobaan x 400 uji")
    print("   n |   KNN | C4.5 | selisih KNN-C4.5 (+- galat baku) | simpul | kedalaman")
    for n in (6, 10, 20, 40, 80, 160, 320):
        g_knn, g_c45, simpul, dalam = [], [], [], []
        for _ in range(60):
            latih = bangkitkan_data(n, rng)
            pohon = bangun_c45(latih)
            s, d = ukuran_c45(pohon)
            simpul.append(s)
            dalam.append(d)
            salah_k = salah_c = 0
            for j in range(400):
                x, y, asli_kelas = titik_uji_acak(j, rng)
                if prediksi_knn(latih, x, y, 3) != asli_kelas:
                    salah_k += 1
                if prediksi_c45(pohon, x, y) != asli_kelas:
                    salah_c += 1
            g_knn.append(100.0 * salah_k / 400)
            g_c45.append(100.0 * salah_c / 400)
        mk = sum(g_knn) / 60
        mc = sum(g_c45) / 60
        selisih = [a - b for a, b in zip(g_knn, g_c45)]
        ms = sum(selisih) / 60
        sd = math.sqrt(sum((v - ms) ** 2 for v in selisih) / 59)
        print(f" {n:3d} | {mk:5.2f} | {mc:4.2f} | {ms:+6.2f} +- {sd / math.sqrt(60):.2f}"
              f"{'':17} | {sum(simpul) / 60:6.1f} | {sum(dalam) / 60:9.1f}")

    # Waktu membangun model dan waktu prediksi per kueri untuk: KNN brute force, KNN dengan
    # KD-Tree, dan C4.5. Perhatikan: C4.5 mahal di pelatihan tetapi murah di prediksi,
    # KNN sebaliknya (tanpa pelatihan, prediksi mahal).
    # [B] waktu pelatihan dan prediksi
    print("\n[B] Waktu (rata-rata 200 kueri)")
    print("    n | latih C4.5 (ms) | bangun KD-Tree (ms) | prediksi per kueri (mikrodetik)")
    print("      |                 |                     |  KNN brute |  KNN KD-Tree |   C4.5")
    for n in (20, 100, 1000, 5000):
        latih = bangkitkan_data(n, rng)
        t = time.perf_counter()
        pohon = bangun_c45(latih)
        t_c45 = (time.perf_counter() - t) * 1000
        titik = [{"id": i, "x": a, "y": b, "kelas": c} for i, (a, b, c) in enumerate(latih)]
        t = time.perf_counter()
        pohon_kd = buat_tree(titik)
        t_kd = (time.perf_counter() - t) * 1000
        kueri = [(rng.uniform(0, 10), rng.uniform(0, 10)) for _ in range(200)]
        t = time.perf_counter()
        for x, y in kueri:
            prediksi_knn(latih, x, y, 3)
        w_brute = (time.perf_counter() - t) / 200 * 1e6
        t = time.perf_counter()
        for x, y in kueri:
            prediksi_kd(pohon_kd, x, y, 3)
        w_kd = (time.perf_counter() - t) / 200 * 1e6
        t = time.perf_counter()
        for x, y in kueri:
            prediksi_c45(pohon, x, y)
        w_c45 = (time.perf_counter() - t) / 200 * 1e6
        print(f" {n:4d} | {t_c45:15.2f} | {t_kd:19.2f} | {w_brute:10.1f} | {w_kd:12.1f} | {w_c45:6.1f}")

    # Contoh aturan yang dihasilkan C4.5 dari satu data latih (bisa langsung dibaca dosen).
    # [C] aturan if-then C4.5 pada satu data latih (n=20) -> mudah dijelaskan
    print("\n[C] Contoh aturan C4.5 dari satu data latih n=20 (dapat dibaca manusia)")
    cetak_aturan(bangun_c45(bangkitkan_data(20, rng)))

    qx, qy, k, kandidat, dikunjungi = asli          # pulihkan nilai program utama

# Dijalankan otomatis di akhir program.
perbandingan_c45()