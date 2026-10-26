# K-NEAREST-NEIGHBOR-and-KD-TREE-
# Repositori Klasifikasi Algoritma Komputasi: Menggunakan Algoritma terapan: optimasi, numerik, data
Program ini mengklasifikasikan buah menjadi Matang atau Belum Matang berdasarkan dua fitur (x, y) menggunakan K-Nearest Neighbor (KNN), mempercepat pencariannya dengan KD-Tree, lalu membandingkannya dengan Decision Tree C4.5. Semua ditulis dari nol tanpa pustaka machine learning.

### Requirements:
1. Python
2. Tidak perlu instal pustaka tambahan (hanya random, math, time)

### Cara menjalankan:
python knn_kdtree_murni.py

Program meminta input:
Masukkan X buah uji (0-10): 5
Masukkan Y buah uji (0-10): 5
Masukkan nilai K (1, 3, atau 5): 3

### Inti program:
KNN (Brute Force)
Menghitung jarak Euclidean dari titik uji ke semua titik latih
Rumus: 
```sh
jarak = √((x1-x2)² + (y1-y2)²) 
```
Urutkan jarak, ambil K terdekat, lalu voting (kelas mayoritas = prediksi)
Jika suara seri (K genap), ikuti tetangga terdekat
Kompleksitas prediksi: 
```sh
O(n log n)
```

### KD-Tree
Pohon biner yang membagi ruang di median, sumbu X dan Y bergantian tiap kedalaman
Pencarian menelusuri sisi dekat dulu, lalu memangkas (pruning) sisi jauh jika |delta| >= jarak kandidat terburuk
Hasilnya sama persis dengan brute force, tetapi simpul yang diperiksa lebih sedikit dari n
Program menampilkan Node dikunjungi: m dari n sebagai bukti penghematan

### Decision Tree C4.5
Dibangun dengan gain ratio (gain / split info), pemisah biner x <= ambang
Hasil berupa aturan JIKA-MAKA yang bisa dibaca manusia
    JIKA x <= 5.35:
       MAKA Belum Matang
    SELAIN ITU (x > 5.35):
       MAKA Matang

### Perbandingan KNN vs Decision Tree C4.5:
Aspek	KNN (+ KD-Tree)	Decision Tree C4.5
Tipe	Lazy (menghafal data)	Eager (merangkum data jadi aturan)
Pelatihan	Tanpa pelatihan (KD-Tree hanya perlu dibangun)	Lebih mahal
Prediksi	Lebih mahal, bergantung pada n	Sangat cepat, hanya sedalam pohon
Interpretasi	Sulit dijelaskan	Mudah, berupa aturan if-then
Galat	Menurun seiring n bertambah	Dibandingkan langsung pada data yang sama

Program membandingkan keduanya pada data yang sama (seed 42) dalam tiga bagian:
[A] Galat klasifikasi untuk berbagai ukuran data n
[B] Waktu pelatihan dan prediksi (brute force vs KD-Tree vs C4.5)
[C] Aturan if-then yang dihasilkan C4.5

### Fitur pendukung:
Kasus uji kecil (8 titik tetap) yang bisa dicek manual, 3 kasus: dekat Belum Matang, dekat Matang, dan suara seri. Tiap kasus dicek LULUS/GAGAL
Analisis galat dan laju konvergensi: galat ≈ n^(-0,48) dan menuju lantai kecil (galat Bayes) akibat daerah tumpang tindih kedua kelas
Tampilan teks: grafik koordinat X-Y, grafik batang jarak, dan struktur KD-Tree (tanda * = dikunjungi, tanpa tanda = dipangkas)
