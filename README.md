# Proyek Akhir: Menyelesaikan Permasalahan Perusahaan Edutech - Alifina Aulia Azzahra


## Business Understanding

Jaya Jaya Institut merupakan salah satu institusi pendidikan perguruan yang telah berdiri sejak tahun 2000. Hingga saat ini ia telah mencetak banyak lulusan dengan reputasi yang sangat baik. Akan tetapi, terdapat banyak juga siswa yang tidak menyelesaikan pendidikannya alias dropout.
Jumlah dropout yang tinggi ini tentunya menjadi salah satu masalah yang besar untuk sebuah institusi pendidikan. Oleh karena itu, Jaya Jaya Institut ingin mengembangkan sistem yang dapat mendeteksi secepat mungkin siswa yang mungkin akan melakukan dropout serta dapat memonitor performa siswa sehingga dapat diberi bimbingan khusus.

### Permasalahan Bisnis  

- Tingginya tingkat dropout siswa yang melebihi 30% di Jaya Jaya Institut menimbulkan kekhawatiran akan efektivitas sistem pembelajaran dan kualitas pelayanan pendidikan.  
- Ketiadaan sistem prediktif dan pemantauan risiko dropout secara real-time menyulitkan institusi dalam melakukan intervensi secara tepat waktu.  
- Kurangnya pemahaman mengenai faktor utama yang menyebabkan siswa dropout, sehingga program intervensi belum optimal dan tidak tepat sasaran.

### Cakupan Proyek  
 
- Mengembangkan model prediksi dropout menggunakan algoritma Logistic Regression, yang mampu mengidentifikasi siswa dengan risiko tinggi untuk keluar dari studi.  
- Membangun aplikasi berbasis Streamlit untuk memudahkan Jaya Jaya Institut dalam melakukan prediksi dropout secara instan berdasarkan data input siswa.  
- Mengembangkan dashboard interaktif untuk visualisasi hasil analisis, yang dapat digunakan Jaya Jaya Institut untuk pemantauan dan pengambilan keputusan berbasis data terhadap risiko dropout siswa.  

#### Resource dan Tools yang Digunakan

- Data performa siswa Jaya Jaya Institut
- Bahasa pemrograman Python untuk pemrosesan data dan pemodelan machine learning.  
- Berbagai library untuk membangun dan mengevaluasi model, proses pembersihan dan manipulasi data, serta pembuatan visualisasi data eksploratif.  
- Streamlit sebagai framework untuk membangun aplikasi web interaktif berbasis Python.  
- Looker Studio untuk membuat dashboard visualisasi interaktif yang menampilkan analisis dropout secara ringkas dan mudah dipahami

### Persiapan

**Sumber data**
 
 Dataset performa siswa dibuat dari sebuah institusi pendidikan tinggi (diperoleh dari beberapa basis data yang terpisah) yang berkaitan dengan siswa yang terdaftar di berbagai program sarjana, seperti agronomi, desain, pendidikan, keperawatan, jurnalistik, manajemen, pelayanan sosial, dan teknologi. Dataset ini mencakup informasi yang diketahui pada saat siswa mendaftar (jalur akademik, demografi, dan faktor sosial-ekonomi) serta kinerja akademik siswa pada akhir semester pertama dan kedua. Dataset tersebut berasal dari [UCML - Predict Students' Dropout and Academic Success](https://archive.ics.uci.edu/dataset/697/predict+students+dropout+and+academic+success) dan terdiri dari 4424 baris dan 37 kolom. Fitur dalam dataset meliputi :

- Marital_status : Status perkawinan siswa
- Application_mode : Mode pendaftaran siswa
- Application_order : Urutan pendaftaran siswa
- Course : Kode program studi yang diikuti siswa
- Daytime_evening_attendance : Jenis kehadiran kuliah (pagi/siang atau malam)
- Previous_qualification : Jenis kualifikasi pendidikan sebelumnya
- Previous_qualification_grade : Nilai kualifikasi pendidikan sebelumnya
- Nacionality : Kebangsaan siswa
- Mothers_qualification : Kualifikasi pendidikan ibu siswa
- Fathers_qualification : Kualifikasi pendidikan ayah siswa
- Mothers_occupation : Pekerjaan ibu siswa
- Fathers_occupation : Pekerjaan ayah siswa
- Admission_grade : Nilai ujian masuk perguruan tinggi
- Displaced : Status pengungsi atau pemindahan tempat tinggal siswa
- Educational_special_needs : Apakah siswa memiliki kebutuhan pendidikan khusus
- Debtor : Status siswa sebagai peminjam atau penunggak biaya
- Tuition_fees_up_to_date : Status pembayaran biaya kuliah (lunas atau belum)
- Gender : Jenis kelamin siswa
- Scholarship_holder : Status penerima beasiswa
- Age_at_enrollment : Usia siswa saat mendaftar
- International : Apakah siswa adalah siswa internasional
- Curricular_units_1st_sem_credited : Jumlah mata kuliah yang diakui pada semester pertama
- Curricular_units_1st_sem_enrolled : Jumlah mata kuliah yang diambil pada semester pertama
- Curricular_units_1st_sem_evaluations : Jumlah evaluasi yang diikuti pada semester pertama
- Curricular_units_1st_sem_approved : Jumlah mata kuliah yang lulus pada semester pertama
- Curricular_units_1st_sem_grade : Nilai rata-rata mata kuliah pada semester pertama
- Curricular_units_1st_sem_without_evaluations : Jumlah mata kuliah tanpa evaluasi pada semester pertama
- Curricular_units_2nd_sem_credited : Jumlah mata kuliah yang diakui pada semester kedua
- Curricular_units_2nd_sem_enrolled : Jumlah mata kuliah yang diambil pada semester kedua
- Curricular_units_2nd_sem_evaluations : Jumlah evaluasi yang diikuti pada semester kedua
- Curricular_units_2nd_sem_approved : Jumlah mata kuliah yang lulus pada semester kedua
- Curricular_units_2nd_sem_grade : Nilai rata-rata mata kuliah pada semester kedua
- Curricular_units_2nd_sem_without_evaluations : Jumlah mata kuliah tanpa evaluasi pada semester kedua
- Unemployment_rate : Tingkat pengangguran di wilayah siswa
- Inflation_rate : Tingkat inflasi di wilayah siswa
- GDP : Produk domestik bruto wilayah siswa
- Status : Status akademik siswa (misal: dropout, lulus, masih aktif)


**Setup Environment**

Project ini menggunakan Python 3.12 dan membutuhkan beberapa pustaka yang telah didefinisikan di dalam file `requirements.txt`.

* **Menggunakan Anaconda**
```
conda create --name main-ds python=3.12
conda activate main-ds
pip install -r requirements.txt
```
* **Menggunakan Pipenv (Shell/Terminal)**
```
mkdir proyek_permasalahan_edutech
cd proyek_permasalahan_edutech
pipenv install
pipenv shell
pip install -r requirements.txt
```


## Data Preprocessing

Sebelum membangun model, dilakukan pembersihan dan transformasi data sebagai berikut:

- **Menangani missing value**: Karena tidak terdapat missing value, maka tidak ada baris yang harus dihapus.
- **Encoding Variabel Kategorikal**: Variabel kategorikal seperti `Status` dikonversi menjadi numerik menggunakan Label Encoding agar dapat digunakan dalam model pembelajaran mesin yang hanya menerima input numerik. Dengan Label Encoding, setiap kategori unik diberi label angka tertentu, misalnya `Dropout` = 0, `Enrolled` = 1, dan `Graduate` = 2.
- **Mendeteksi Outlier**: Terdapat outlier pada beberapa kolom seperti `Scholarship_holder`, `Curricular_units_2nd_sem_grade`, `Curricular_units_1st_sem_grade`, `Previous_qualification`, `Curricular_units_1st_sem_credited`, dan masih banyak lagi. Akan tetapi, karena  merepresentasikan nilai asli, maka outlier akan dibiarkan tanpa dihapus atau diimputasi.
- **Standardisasi Fitur**: Dengan mengubah semua fitur menjadi distribusi dengan rata-rata 0 dan standar deviasi 1, StandardScaler membantu memastikan bahwa model berfokus pada hubungan antar variabel, bukan skala absolutnya, serta mempercepat proses konvergensi saat pelatihan.
- **Mengubah klasifikasi multiclass menjadi biner** : Target variabel `Status` dikonversi menjadi format biner, dengan nilai 1 untuk siswa yang dropout dan 0 untuk siswa yang enrolled maupun graduate, agar dapat digunakan dalam klasifikasi biner. Langkah ini dipilih karena klasifikasi biner lebih efektif dalam konteks prediksi dropout, dimana fokus utama adalah membedakan antara siswa yang berisiko berhenti dan yang tidak. 


## Modeling

### Pemilihan Fitur Menggunakan Random Forest

Sebelum membangun model prediksi, dilakukan seleksi fitur menggunakan algoritma Random Forest. Random Forest adalah metode *ensemble learning* berbasis banyak decision tree yang juga mampu memberikan informasi mengenai feature importance, yaitu seberapa besar kontribusi setiap fitur dalam proses prediksi. Keunggulan Random Forest dalam konteks ini adalah:

- Kemampuan menangani data numerik dan kategorikal.
- Robust terhadap overfitting, karena setiap pohon dilatih dengan subset acak dari data.
- Memberikan informasi feature importance, yang sangat berguna untuk seleksi fitur.

Berdasarkan hasil perhitungan feature importance, dipilih 10 fitur teratas yang paling berpengaruh terhadap prediksi dropout siswa. Fitur-fitur ini kemudian digunakan dalam proses pelatihan model selanjutnya. Fitur-fitur tersebut diantaranya yaitu :
   - Curricular_units_2nd_sem_approved
   - Curricular_units_2nd_sem_grade
   - Curricular_units_1st_sem_approved
   - Tuition_fees_up_to_date
   - Curricular_units_1st_sem_grade
   - Age_at_enrollment
   - Admission_grade
   - Previous_qualification_grade
   - Curricular_units_2nd_sem_evaluations
   - Course

---

### Pembangunan Model dengan Logistic Regression

Setelah fitur terbaik diperoleh dari Random Forest, proses pembangunan model prediksi dilakukan menggunakan **Logistic Regression**. 

#### Alasan Pemilihan Logistic Regression:

- Model yang lebih sederhana dan transparan, sehingga memudahkan interpretasi (misalnya pengaruh arah dan kekuatan dari masing-masing fitur).
- Efisien secara komputasi, terutama untuk dataset yang tidak terlalu besar.
- Sangat cocok untuk kasus klasifikasi biner, seperti memprediksi apakah seorang siswa akan dropout atau tidak.
- Lebih mudah ditindaklanjuti oleh pihak akademik, karena koefisien regresi dapat dihubungkan dengan faktor risiko.

---

### Langkah-Langkah Pembangunan Model

1. **Seleksi Fitur**  
   Dari hasil perhitungan feature importance oleh Random Forest, dipilih 10 fitur terbaik yang digunakan sebagai input dalam proses modeling.

2. **Pisahkan Fitur dan Target**  
   Target prediksi adalah variabel `status_encoded` yang dikonversi menjadi biner:  
   - 1 untuk siswa dropout,  
   - 0 untuk siswa yang enrolled maupun graduate.

3. **Pembagian Data (Train-Test Split)**  
   Data dibagi menjadi 80% data pelatihan dan 20% data pengujian. Digunakan parameter `stratify` untuk menjaga distribusi kelas tetap seimbang pada kedua bagian data.

4. **Standardisasi Fitur**  
   Karena Logistic Regression sensitif terhadap skala data, dilakukan standardisasi terhadap fitur numerik menggunakan `StandardScaler`.

5. **Pelatihan Model Logistic Regression**  
   Model Logistic Regression dilatih menggunakan data pelatihan yang telah diskalakan.

6. **Evaluasi Model**  
   Model kemudian diuji pada data pengujian. Evaluasi dilakukan menggunakan:
   - **Classification Report**: untuk melihat precision, recall, dan f1-score masing-masing kelas.
   - **Confusion Matrix**: untuk melihat jumlah prediksi benar dan salah dari tiap kelas.

---

### Hasil Evaluasi

Berdasarkan hasil dari classification report dan confusion matrix, model **Logistic Regression** menunjukkan **akurasi keseluruhan sebesar 87%**. 

- Untuk kelas 0 (tidak dropout):  
  - Precision: **0.87**  
  - Recall: **0.96**  
  - F1-score: **0.91**  
  Model sangat baik dalam mengidentifikasi siswa yang tetap melanjutkan studi.

- Untuk kelas 1 (dropout):  
  - Precision: **0.88**  
  - Recall: **0.69**  
  - F1-score: **0.77**  
  Meskipun performa prediksi dropout cukup baik, recall yang masih rendah menunjukkan potensi siswa dropout yang tidak terdeteksi oleh model.

Model ini dapat menjadi alat bantu awal dalam sistem peringatan dini (early warning system), namun dapat ditingkatkan lebih lanjut dengan pendekatan balancing, tuning threshold, atau metode klasifikasi lainnya.


## Business Dashboard

Dashboard **"Student Performance and Dropout Monitoring"** dirancang untuk membantu pihak akademik dan manajemen Jaya Jaya Institut dalam memantau performa siswa serta memahami pola dan faktor penyebab dropout. Dashboard ini menyajikan visualisasi utama sebagai berikut:

- **Statistik Umum**:  
  - Total siswa: 4.424  
  - Total Dropout: 1.421  
  - Tingkat Dropout: 32.12%  
  - Rata-rata Nilai Masuk: 126.98  
  - Jumlah Mata Kuliah: 17

- **Visualisasi Utama**:
  - **Proporsi Dropout Berdasarkan Gender**: Proporsi hampir seimbang antara pria (50.7%) dan wanita (49.3%), menunjukkan tidak ada perbedaan signifikan secara gender.
  - **Dropout Berdasarkan Usia**: Sebagian besar siswa yang dropout berada pada kategori usia 17–30 tahun (1.036 siswa), menunjukkan bahwa usia muda merupakan kelompok paling rentan.
  - **Dropout Berdasarkan Status Pembayaran Uang Kuliah**: Sebagian besar dropout berasal dari siswa yang belum membayar uang kuliah tepat waktu (457 dari 528 siswa dalam kategori ini tidak bayar).
  - **Distribusi Siswa Berdasarkan Nilai Semester 2 dan Status**: Siswa dengan nilai rendah (0–10) didominasi oleh status dropout, sedangkan nilai tinggi (16 ke atas) didominasi oleh siswa lulus.
  - **Rata-rata Mata Kuliah Disetujui per Semester**: Siswa yang dropout memiliki rata-rata mata kuliah yang disetujui jauh lebih rendah (Semester 1: 2.6, Semester 2: 1.9) dibandingkan siswa lulus (Semester 1 & 2: 6.2).
  - **Tingkat Dropout Berdasarkan Mata Kuliah**:  
    - *Biofuel Production Technology* memiliki tingkat dropout tertinggi (66.7%)  
    - Diikuti oleh *Equiculture* (55.3%) dan *Informatics Engineering* (54.1%)

Dashboard ini dilengkapi dengan filter interaktif berdasarkan **kewarganegaraan (Nationality)** yang memungkinkan pengguna untuk melakukan analisis mendalam berdasarkan latar belakang siswa secara geografis.


**Akses Dashboard**: [Student Performance and Dropout Monitoring Dashboard di Looker Studio](https://lookerstudio.google.com/reporting/9fbe1c13-8c3c-4516-a6c9-9bc6bdcb9dc1)


## Menjalankan Sistem Machine Learning

Untuk menjalankan prototype sistem machine learning prediksi dropout yang telah dibuat menggunakan Streamlit, ikuti langkah-langkah berikut:

1. **Persiapkan lingkungan kerja**  
   Pastikan Python dan semua dependensi yang dibutuhkan sudah terinstall. Jika belum, bisa menginstall dependensi dengan menjalankan:  
   ```
   pip install -r requirements.txt
   ```
2. **Jalankan aplikasi Streamlit secara lokal**  
   Buka terminal atau command prompt, navigasikan ke folder yang berisi file `app.py`, lalu jalankan perintah berikut:
   ```
   streamlit run app.py
   ```  
   Perintah ini akan menjalankan aplikasi dan biasanya membuka halaman web prototype secara otomatis di browser.
3. **Gunakan versi prototype yang sudah dideploy**  
   Jika tidak ingin menjalankan secara lokal, pengguna juga dapat langsung mengakses aplikasi prototype melalui tautan berikut:
   [Aplikasi Student Dropout Risk Prediction (Streamlit)](https://proyekpermasalahanedutech-alifinaaa.streamlit.app/)
   
   Versi ini dapat digunakan untuk mencoba aplikasi tanpa harus menginstal apapun di perangkat lokal.

## Conclusion

Berdasarkan analisis data siswa dan pengembangan dashboard **"Student Performance and Dropout Monitoring"**, diperoleh beberapa kesimpulan utama sebagai berikut:

1. Berdasarkan analisis feature importance menggunakan model klasifikasi dan pengamatan visualisasi data, ditemukan bahwa 10 faktor utama yang paling memengaruhi kemungkinan siswa mengalami dropout adalah:
   - **Curricular_units_2nd_sem_approved**: Siswa yang gagal menyelesaikan mata kuliah di semester 2 cenderung memiliki risiko dropout lebih tinggi.
   - **Curricular_units_2nd_sem_grade**: Nilai rendah pada semester 2 menjadi indikator kuat potensi ketidaklulusan.
   - **Curricular_units_1st_sem_approved**: Jumlah mata kuliah semester 1 yang diselesaikan juga menjadi penentu penting terhadap kelulusan.
   - **Tuition_fees_up_to_date**: Siswa yang tidak membayar uang kuliah secara tepat waktu memiliki kemungkinan besar untuk dropout.
   - **Curricular_units_1st_sem_grade**: Nilai akademik semester 1 yang rendah mengindikasikan awal dari potensi kegagalan studi.
   - **Age_at_enrollment**: Usia saat mendaftar memengaruhi performa belajar, usia terlalu muda atau terlalu tua memiliki tantangan tersendiri.
   - **Admission_grade**: Nilai masuk yang rendah menunjukkan kesiapan akademik awal yang minim, meningkatkan risiko dropout.
   - **Previous_qualification_grade**: Latar belakang pendidikan sebelumnya juga berkontribusi dalam membentuk kesiapan belajar siswa.
   - **Curricular_units_2nd_sem_evaluations**: Banyaknya evaluasi di semester 2 menunjukkan beban akademik yang berpotensi memengaruhi performa.
   - **Course**: Program studi tertentu memiliki tingkat dropout yang lebih tinggi, menandakan perlunya evaluasi kurikulum atau dukungan akademik.

2. Untuk memprediksi secara dini siswa yang berisiko tidak lulus tepat waktu, institusi dapat menggunakan model prediktif berbasis data, seperti yang dilakukan dengan Logistic Regression. Model ini, setelah dilatih dengan data historis, dapat memberikan prediksi yang cukup akurat mengenai kemungkinan status kelulusan siswa berdasarkan fitur-fitur penting yang telah diidentifikasi, dengan tingkat akurasi yang memadai yaitu 87%.

3.  Dashboard yang dikembangkan menyajikan data interaktif untuk menggambarkan hubungan antara berbagai variabel dengan status kelulusan siswa. Visualisasi ini mencakup distribusi nilai, jumlah mata kuliah yang disetujui, serta status pembayaran uang kuliah yang membantu mengidentifikasi pola dropout sejak semester awal.

Secara keseluruhan, dengan membangun model prediktif yang efektif dan memanfaatkan analisis data, Jaya Jaya Institut dapat mengidentifikasi siswa yang berisiko tidak lulus tepat waktu lebih awal, sehingga dapat memberikan intervensi dan dukungan yang tepat sasaran untuk meningkatkan keberhasilan akademik. Hal ini akan membantu meningkatkan efisiensi proses pembinaan siswa, memperbaiki tingkat kelulusan tepat waktu, serta meningkatkan reputasi institusi melalui peningkatan kualitas pendidikan dan kepuasan siswa secara keseluruhan.


### Rekomendasi Action Items

- **Pendampingan Akademik untuk Siswa dengan Nilai Rendah**  
  Berikan bimbingan khusus bagi siswa yang memiliki nilai rendah pada mata kuliah semester 1 dan 2 untuk meningkatkan performa akademik mereka.

- **Monitoring Pembayaran UKT Tepat Waktu**  
  Pantau dan bantu siswa yang belum melunasi biaya kuliah tepat waktu dengan memberikan opsi pembayaran yang lebih fleksibel dan pengingat berkala.

- **Pemanfaatan Data Nilai dan Kualifikasi Awal untuk Seleksi dan Pembinaan**  
  Gunakan data nilai penerimaan dan kualifikasi sebelumnya untuk mengidentifikasi siswa yang membutuhkan perhatian ekstra sejak awal.

- **Penggunaan Dashboard Interaktif untuk Monitoring Berkala**  
  Integrasikan dashboard untuk memantau perkembangan akademik siswa secara real-time dan lakukan intervensi tepat waktu.

- **Edukasi Tentang Pentingnya Penyelesaian Mata Kuliah Tepat Waktu**  
  Sosialisasikan dampak jumlah mata kuliah yang diselesaikan pada semester 1 dan 2 terhadap kelulusan agar siswa lebih sadar dalam merencanakan studinya.
