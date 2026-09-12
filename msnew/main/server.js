const express = require('express');
const path = require('path');
const app = express();

app.set('view engine', 'ejs');
app.set('views', path.join(__dirname, 'views'));

// Supaya bisa baca file CSS yang ada di folder 'ms/public' secara bersamaan
app.use(express.static(path.join(__dirname, '../ms/public')));
app.use(express.urlencoded({ extended: true }));

// Halaman Utama / Login
app.get('/', (req, res) => {
    res.render('login', { title: 'Login - PT Membahas Saham Indonesia' });
});

// Proses saat user klik tombol Login
app.post('/auth/login', (req, res) => {
    const { email, password } = req.body;
    // Nanti proses cek database di sini. 
    // Untuk sekarang, kalau sukses langsung diarahkan ke subdomain VIP:
    res.redirect('https://vip.membahassahamindonesia.com');
});

// Jalankan di port yang berbeda (misal port 3000)
const PORT = 3000;
app.listen(PORT, () => {
    console.log(`Server Utama berjalan di http://localhost:${PORT}`);
});