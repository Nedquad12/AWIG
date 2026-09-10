require('dotenv').config();
const express = require('express');
const cors = require('cors');
const path = require('path');
const fs = require('fs');
const multer = require('multer');
const FormData = require('form-data');
const axios = require('axios'); // Menggunakan axios menggantikan fetch

const app = express();

// Buat folder uploads jika belum ada
const uploadDir = path.join(__dirname, 'public', 'uploads');
if (!fs.existsSync(uploadDir)) {
    fs.mkdirSync(uploadDir, { recursive: true });
}

// Konfigurasi Multer
const storage = multer.diskStorage({
    destination: (req, file, cb) => cb(null, uploadDir),
    filename: (req, file, cb) => {
        const uniqueSuffix = Date.now() + '-' + Math.round(Math.random() * 1E9);
        const ext = path.extname(file.originalname);
        cb(null, `proof-${uniqueSuffix}${ext}`);
    }
});
const upload = multer({ storage: storage });

app.set('views', path.join(__dirname, 'views'));
app.set('view engine', 'ejs');
app.use(express.static(path.join(__dirname, 'public')));

app.use(cors());
app.use(express.json());
app.use(express.urlencoded({ extended: true }));

// CONFIG TELEGRAM BOT
const BOT_TOKEN = process.env.BOT_TOKEN || "8946137331:AAFfLuDhdClNIiSUdSZOvCDg7MenXhr6HDE";
const ADMIN_GROUP_ID = process.env.ADMIN_GROUP_ID || "-1003758450134";
const VIP_GROUP_IDS = [-1002738891883, -1004422072210, -1004466109703];

// Helper Sleep untuk jeda request
const sleep = (ms) => new Promise((resolve) => setTimeout(resolve, ms));

// Helper Request Telegram dengan Auto-Retry jika Kena Rate Limit 429
async function safeTelegramRequest(requestFn, maxRetries = 3) {
    for (let attempt = 0; attempt < maxRetries; attempt++) {
        try {
            return await requestFn();
        } catch (err) {
            const errData = err.response?.data;
            if (errData?.error_code === 429 && errData?.parameters?.retry_after) {
                const waitSec = errData.parameters.retry_after + 1;
                console.log(`[Rate Limit 429] Menunggu ${waitSec} detik sebelum mencoba lagi...`);
                await sleep(waitSec * 1000);
            } else {
                throw err;
            }
        }
    }
    throw new Error('Gagal mengirim request ke Telegram setelah beberapa kali percobaan.');
}

// Helper: Generate VIP Invite Links via Axios dengan Delay
async function createVipInviteLinks() {
    let inviteLinksText = "";
    const expireDate = Math.floor(Date.now() / 1000) + (24 * 60 * 60);

    for (const groupId of VIP_GROUP_IDS) {
        try {
            await sleep(500); // Jeda 500ms antar request

            // Get Chat Info
            const chatRes = await safeTelegramRequest(() => 
                axios.get(`https://api.telegram.org/bot${BOT_TOKEN}/getChat?chat_id=${groupId}`)
            );
            const groupTitle = chatRes.data.ok ? chatRes.data.result.title : `Grup ${groupId}`;

            await sleep(500); // Jeda 500ms

            // Create Invite Link
            const linkRes = await safeTelegramRequest(() => 
                axios.post(`https://api.telegram.org/bot${BOT_TOKEN}/createChatInviteLink`, {
                    chat_id: groupId,
                    expire_date: expireDate,
                    member_limit: 1
                })
            );

            if (linkRes.data.ok) {
                inviteLinksText += `• <b>${groupTitle}:</b>\n${linkRes.data.result.invite_link}\n\n`;
            }
        } catch (err) {
            console.error(`Gagal membuat link untuk ${groupId}:`, err.response?.data || err.message);
        }
    }
    return inviteLinksText;
}

// API SUBMIT MANUAL PAYMENT
app.post('/api/submit-manual-payment', upload.single('paymentProof'), async (req, res) => {
    try {
        const { packageCode, name, email, telegramUsername, promoCode, finalPrice, senderBank, senderAccountNumber } = req.body;

        if (!req.file) return res.status(400).json({ status: 'error', message: 'File bukti pembayaran wajib diunggah.' });

        const selectedPackage = getPackageByCode(packageCode);
        if (!selectedPackage) return res.status(400).json({ status: 'error', message: 'Kode paket tidak valid.' });

        const orderId = `MSI-WEB-${Date.now()}`;
        const proofFilePath = req.file.path;

        // 1. Generate VIP Invite Links (dengan safe request)
        const generatedLinks = await createVipInviteLinks();

        // 2. Format Teks untuk Grup Admin
        const captionAdmin = 
            `📥 <b>TRANSAKSI WEB BARU (OTOMATIS)</b>\n\n` +
            `🆔 <b>Order ID:</b> <code>${orderId}</code>\n` +
            `👤 <b>Nama:</b> ${name}\n` +
            `📧 <b>Email:</b> ${email}\n` +
            `📱 <b>Telegram:</b> ${telegramUsername.startsWith('@') ? telegramUsername : '@' + telegramUsername}\n` +
            `📦 <b>Paket:</b> ${selectedPackage.duration} (${packageCode})\n` +
            `🏷️ <b>Promo:</b> ${promoCode || '-'}\n` +
            `💰 <b>Total Bayar:</b> Rp ${parseInt(finalPrice).toLocaleString('id-ID')}\n` +
            `🏦 <b>Bank/E-Wallet Pengirim:</b> ${senderBank || '-'} (${senderAccountNumber || '-'})\n\n` +
            `🔗 <b>LINK INVITE VIP (BERLAKU 24 JAM / 1 ORANG):</b>\n` +
            `${generatedLinks || 'Gagal generate link otomatis.'}`;

        await sleep(500); // Jeda sebelum kirim foto

        // 3. Kirim Foto Bukti via Safe Telegram Request
        await safeTelegramRequest(async () => {
            const formData = new FormData();
            formData.append('chat_id', ADMIN_GROUP_ID);
            formData.append('photo', fs.createReadStream(proofFilePath));
            formData.append('caption', captionAdmin);
            formData.append('parse_mode', 'HTML');

            return await axios.post(`https://api.telegram.org/bot${BOT_TOKEN}/sendPhoto`, formData, {
                headers: formData.getHeaders()
            });
        });

        return res.status(200).json({
            status: 'success',
            message: 'Bukti pembayaran diterima.',
            orderId,
            email
        });

    } catch (error) {
        console.error('Submit Payment Error:', error.response?.data || error.message);
        return res.status(500).json({ status: 'error', message: 'Gagal memproses bukti pembayaran.' });
    }
});

// Master Data Harga
const PRICING_DATA = {
    ihsg: [
        { code: 'ihsg_1m', duration: '1 Bulan', price: 199000, priceFormatted: 'Rp 199.000', save: null, desc: 'Trial & Evaluasi Fitur', popular: false },
        { code: 'ihsg_3m', duration: '3 Bulan', price: 555000, priceFormatted: 'Rp 555.000', save: 'Hemat Rp 42.000', desc: 'Pendampingan 1 Kuartal', popular: false },
        { code: 'ihsg_6m', duration: '6 Bulan', price: 1080000, priceFormatted: 'Rp 1.080.000', save: 'Hemat Rp 114.000', desc: 'Paket Favorit Member', popular: false },
        { code: 'ihsg_12m', duration: '1 Tahun', price: 1800000, priceFormatted: 'Rp 1.800.000', save: 'Hemat Rp 588.000', desc: 'Investasi Belajar Maksimal', popular: true, badge: 'Paling Hemat' }
    ],
    multiAsset: [
        { code: 'all_1m', duration: '1 Bulan', price: 299000, priceFormatted: 'Rp 299.000', save: null, desc: 'Akses Semua Aset', popular: false },
        { code: 'all_3m', duration: '3 Bulan', price: 855000, priceFormatted: 'Rp 855.000', save: 'Hemat Rp 42.000', desc: 'Multi-Asset Kuartalan', popular: false },
        { code: 'all_6m', duration: '6 Bulan', price: 1620000, priceFormatted: 'Rp 1.620.000', save: 'Hemat Rp 174.000', desc: 'Pendampingan 6 Bulan', popular: false },
        { code: 'all_12m', duration: '12 Bulan', price: 3000000, priceFormatted: 'Rp 3.000.000', save: 'Hemat Rp 588.000', desc: 'Akses Penuh 1 Tahun', popular: true, badge: 'Best Value' }
    ]
};

function getPackageByCode(code) {
    const allPackages = [...PRICING_DATA.ihsg, ...PRICING_DATA.multiAsset];
    return allPackages.find(p => p.code === code) || null;
}

function getPromos() {
    try {
        const promoPath = path.join(__dirname, 'promos.json');
        if (fs.existsSync(promoPath)) {
            const rawData = fs.readFileSync(promoPath);
            return JSON.parse(rawData);
        }
    } catch (err) {
        console.error("Gagal membaca file promos.json:", err);
    }
    return {};
}

// ROUTES
app.get('/', (req, res) => res.render('index', { title: 'MEMBAHAS SAHAM INDONESIA - Official VIP Portal', pricing: PRICING_DATA }));
app.get('/faq', (req, res) => res.render('faq', { title: 'F.A.Q - PT Membahas Saham Indonesia' }));
app.get('/contact', (req, res) => res.render('contact', { title: 'Contact Us - PT Membahas Saham Indonesia' }));
app.get('/checkout', (req, res) => {
    const packageCode = req.query.package;
    const selectedPackage = getPackageByCode(packageCode);
    if (!selectedPackage) return res.redirect('/#pricing-section');
    res.render('checkout', { title: 'Checkout VIP Member - PT Membahas Saham Indonesia', selectedPackage });
});
app.get('/success', (req, res) => res.render('success', { title: 'Status Pembayaran - PT Membahas Saham Indonesia' }));

// API VERIFY PROMO
app.post('/api/verify-promo', (req, res) => {
    const { promoCode, packageCode } = req.body;
    const promos = getPromos();
    const cleanCode = (promoCode || '').toUpperCase().trim();
    const selectedPackage = getPackageByCode(packageCode);

    if (!selectedPackage) return res.status(400).json({ status: 'error', message: 'Paket tidak ditemukan.' });

    const promo = promos[cleanCode];
    if (!promo) return res.status(400).json({ status: 'error', message: 'Kode promo tidak valid atau kedaluwarsa.' });

    let discountAmount = promo.type === 'percentage' ? Math.round((selectedPackage.price * promo.value) / 100) : promo.value;
    const finalPrice = Math.max(0, selectedPackage.price - discountAmount);

    return res.json({
        status: 'success',
        description: promo.description,
        originalPrice: selectedPackage.price,
        discountAmount,
        finalPrice
    });
});

// API SUBMIT MANUAL PAYMENT
app.post('/api/submit-manual-payment', upload.single('paymentProof'), async (req, res) => {
    try {
        const { packageCode, name, email, telegramUsername, promoCode, finalPrice, senderBank, senderAccountNumber } = req.body;

        if (!req.file) return res.status(400).json({ status: 'error', message: 'File bukti pembayaran wajib diunggah.' });

        const selectedPackage = getPackageByCode(packageCode);
        if (!selectedPackage) return res.status(400).json({ status: 'error', message: 'Kode paket tidak valid.' });

        const orderId = `MSI-WEB-${Date.now()}`;
        const proofFilePath = req.file.path;

        // 1. Generate VIP Invite Links
        const generatedLinks = await createVipInviteLinks();

        // 2. Format Teks untuk Grup Admin
        const captionAdmin = 
            `📥 <b>TRANSAKSI WEB BARU (OTOMATIS)</b>\n\n` +
            `🆔 <b>Order ID:</b> <code>${orderId}</code>\n` +
            `👤 <b>Nama:</b> ${name}\n` +
            `📧 <b>Email:</b> ${email}\n` +
            `📱 <b>Telegram:</b> ${telegramUsername.startsWith('@') ? telegramUsername : '@' + telegramUsername}\n` +
            `📦 <b>Paket:</b> ${selectedPackage.duration} (${packageCode})\n` +
            `🏷️ <b>Promo:</b> ${promoCode || '-'}\n` +
            `💰 <b>Total Bayar:</b> Rp ${parseInt(finalPrice).toLocaleString('id-ID')}\n` +
            `🏦 <b>Bank/E-Wallet Pengirim:</b> ${senderBank || '-'} (${senderAccountNumber || '-'})\n\n` +
            `🔗 <b>LINK INVITE VIP (BERLAKU 24 JAM / 1 ORANG):</b>\n` +
            `${generatedLinks || 'Gagal generate link otomatis.'}`;

        // 3. Kirim Foto Bukti via Axios + FormData
        const formData = new FormData();
        formData.append('chat_id', ADMIN_GROUP_ID);
        formData.append('photo', fs.createReadStream(proofFilePath));
        formData.append('caption', captionAdmin);
        formData.append('parse_mode', 'HTML');

        await axios.post(`https://api.telegram.org/bot${BOT_TOKEN}/sendPhoto`, formData, {
            headers: formData.getHeaders()
        });

        return res.status(200).json({
            status: 'success',
            message: 'Bukti pembayaran diterima.',
            orderId,
            email
        });

    } catch (error) {
        console.error('Submit Payment Error:', error.response?.data || error.message);
        return res.status(500).json({ status: 'error', message: 'Gagal memproses bukti pembayaran.' });
    }
});

const PORT = process.env.PORT || 3000;
app.listen(PORT, () => {
    console.log(`🚀 Server aktif di http://localhost:${PORT}`);
});