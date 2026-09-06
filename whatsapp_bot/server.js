const express = require('express');
const { Client, LocalAuth, MessageMedia } = require('whatsapp-web.js');
const qrcode = require('qrcode-terminal');
const fs = require('fs');
const path = require('path');
const os = require('os');

const app = express();
app.use(express.json({ limit: '50mb' }));

const PORT = process.env.PORT || 3000;

// Dynamic Chrome / Chromium / Edge executable resolution on Windows, Linux & macOS
const userHome = os.homedir() || process.env.USERPROFILE || process.env.HOME || '';
const localAppData = process.env.LOCALAPPDATA || (userHome ? path.join(userHome, 'AppData', 'Local') : '');
const programFiles = process.env.PROGRAMFILES || 'C:\\Program Files';
const programFilesX86 = process.env['PROGRAMFILES(X86)'] || 'C:\\Program Files (x86)';

const possibleChromePaths = [
    path.join(localAppData, 'Google', 'Chrome', 'Application', 'chrome.exe'),
    path.join(programFiles, 'Google', 'Chrome', 'Application', 'chrome.exe'),
    path.join(programFilesX86, 'Google', 'Chrome', 'Application', 'chrome.exe'),
    path.join(programFiles, 'Microsoft', 'Edge', 'Application', 'msedge.exe'),
    path.join(programFilesX86, 'Microsoft', 'Edge', 'Application', 'msedge.exe'),
    '/usr/bin/google-chrome',
    '/usr/bin/chromium',
    '/usr/bin/chromium-browser',
    '/Applications/Google Chrome.app/Contents/MacOS/Google Chrome',
    '/Applications/Microsoft Edge.app/Contents/MacOS/Microsoft Edge'
];

let executablePath = possibleChromePaths.find(p => p && fs.existsSync(p));

if (executablePath) {
    console.log(`[WHATSAPP BOT] Using local browser executable: ${executablePath}`);
} else {
    console.log('[WHATSAPP BOT] Local Chrome/Edge not found, falling back to default Puppeteer Chromium.');
}

// Bot State
let latestQr = null;
let isReady = false;
let clientInfo = null;

// Initialize WhatsApp client
function createClient() {
    const client = new Client({
        authStrategy: new LocalAuth({
            dataPath: './session'
        }),
        authTimeoutMs: 90000,
        qrTimeoutMs: 90000,
        puppeteer: {
            headless: true,
            executablePath: executablePath,
            protocolTimeout: 90000,
            args: [
                '--no-sandbox',
                '--disable-setuid-sandbox',
                '--disable-dev-shm-usage',
                '--disable-gpu',
                '--no-first-run',
                '--no-zygote',
                '--disable-background-timer-throttling',
                '--disable-backgrounding-occluded-windows',
                '--disable-renderer-backgrounding',
                '--disable-breakpad',
                '--disable-component-extensions-with-background-pages',
                '--disable-extensions',
                '--mute-audio',
                '--no-default-browser-check',
                '--disable-ipc-flooding-protection'
            ]
        }
    });

    client.on('qr', (qr) => {
        latestQr = qr;
        isReady = false;
        console.log('\n==================================================================');
        console.log('SCAN THIS QR CODE WITH YOUR WHATSAPP MOBILE APP (LINK A DEVICE):');
        console.log('Or open http://localhost:3000/qr in your browser to view QR Code');
        console.log('==================================================================\n');
        qrcode.generate(qr, { small: true });
    });

    client.on('ready', () => {
        isReady = true;
        latestQr = null;
        clientInfo = client.info;
        console.log('\n[WHATSAPP BOT] ✅ WhatsApp Client is READY and logged in!');
        if (clientInfo && clientInfo.wid) {
            console.log(`[WHATSAPP BOT] Logged in as: ${clientInfo.wid.user}`);
        }
    });

    client.on('authenticated', () => {
        console.log('[WHATSAPP BOT] Authentication successful!');
    });

    client.on('auth_failure', (msg) => {
        isReady = false;
        console.error('[WHATSAPP BOT] Authentication failure:', msg);
    });

    client.on('disconnected', async (reason) => {
        isReady = false;
        latestQr = null;
        console.log('[WHATSAPP BOT] Client was logged out / disconnected. Reason:', reason);
        if (reason === 'LOGOUT' || reason === 'NAVIGATION') {
            console.log('[WHATSAPP BOT] Expired/logged-out session detected. Auto-restarting with clean session for fresh QR code...');
            setTimeout(async () => {
                try {
                    await restartClient(true);
                } catch (e) {
                    console.error('[WHATSAPP BOT] Auto-restart error:', e.message);
                }
            }, 1000);
        }
    });

    return client;
}

let client = createClient();

// Helper to restart / reset client
async function restartClient(cleanSession = false) {
    try {
        console.log('[WHATSAPP BOT] Destroying existing client instance...');
        try {
            await client.destroy();
        } catch (e) {
            console.warn('[WHATSAPP BOT] Warning on destroy:', e.message);
        }
        isReady = false;
        latestQr = null;
        clientInfo = null;

        if (cleanSession) {
            console.log('[WHATSAPP BOT] Cleaning session folder...');
            const sessionPath = path.join(__dirname, 'session');
            if (fs.existsSync(sessionPath)) {
                fs.rmSync(sessionPath, { recursive: true, force: true });
            }
            const authPath = path.join(__dirname, '.wwebjs_auth');
            if (fs.existsSync(authPath)) {
                fs.rmSync(authPath, { recursive: true, force: true });
            }
        }

        console.log('[WHATSAPP BOT] Re-initializing WhatsApp client...');
        client = createClient();
        client.initialize();
        return true;
    } catch (err) {
        console.error('[WHATSAPP BOT] Error restarting client:', err);
        throw err;
    }
}

// POST /send endpoint to send alerts with sub-second response
app.post('/send', async (req, res) => {
    const t0 = Date.now();
    const { to, message, image, immediate_ack, send_text_first } = req.body;

    if (!to || !message) {
        return res.status(400).json({ error: 'Missing "to" or "message" in request body.' });
    }

    if (!isReady) {
        return res.status(503).json({ error: 'WhatsApp client is not ready. Please scan QR code first.' });
    }

    let formattedNum = to.replace(/[^0-9]/g, '');
    if (!formattedNum.endsWith('@c.us')) {
        formattedNum = `${formattedNum}@c.us`;
    }

    // Function to perform the actual send (strictly 1 single consolidated message containing all details)
    const doSend = async () => {
        if (image) {
            let base64Data = image;
            if (image.includes(',')) {
                base64Data = image.split(',')[1];
            }
            const media = new MessageMedia('image/jpeg', base64Data, 'threat_alert.jpg');
            // Send exactly 1 single media message with all incident details in the caption
            await client.sendMessage(formattedNum, media, { caption: message, sendMediaAsDocument: false });
            console.log(`[WHATSAPP BOT] Single consolidated media alert delivered to ${formattedNum} in ${Date.now() - t0}ms`);
        } else {
            // Text-only single message
            await client.sendMessage(formattedNum, message);
            console.log(`[WHATSAPP BOT] Single text alert delivered to ${formattedNum} in ${Date.now() - t0}ms`);
        }
    };

    if (immediate_ack) {
        // Return 200 immediately so HTTP caller unblocks in < 5ms
        res.status(200).json({ success: true, status: 'QUEUED', message: 'Alert queued for immediate background dispatch.' });
        doSend().catch(err => console.error('[WHATSAPP BOT] Background send error:', err.message));
    } else {
        try {
            await doSend();
            res.status(200).json({ success: true, duration_ms: Date.now() - t0, message: 'Alert message sent successfully.' });
        } catch (err) {
            console.error('[WHATSAPP BOT] Error sending message:', err);
            res.status(500).json({ error: err.message, duration_ms: Date.now() - t0 });
        }
    }
});

// GET /status
app.get('/status', (req, res) => {
    res.json({
        ready: isReady,
        hasQr: !!latestQr,
        user: clientInfo ? clientInfo.wid.user : null
    });
});

// GET & POST /refresh - refresh connection without deleting saved session
app.all('/refresh', async (req, res) => {
    try {
        await restartClient(false);
        res.json({ success: true, message: 'WhatsApp client refreshed successfully.' });
    } catch (err) {
        res.status(500).json({ error: err.message });
    }
});

// GET & POST /reset - clear session cache & force new QR code
app.all('/reset', async (req, res) => {
    try {
        await restartClient(true);
        res.json({ success: true, message: 'WhatsApp session reset completely. Scan new QR code.' });
    } catch (err) {
        res.status(500).json({ error: err.message });
    }
});

// GET /qr - HTML page rendering QR code for easy scanning in browser
app.get('/qr', (req, res) => {
    if (isReady) {
        return res.send(`
            <html>
                <body style="font-family:sans-serif; text-align:center; padding:50px; background:#121212; color:#fff;">
                    <h1 style="color:#39ff14;">✅ WhatsApp Bot is Ready & Logged In!</h1>
                    <p>Logged in user: <b>${clientInfo ? clientInfo.wid.user : 'Connected'}</b></p>
                    <a href="/reset" style="color:#ff4444;">Reset Session & Scan New QR</a>
                </body>
            </html>
        `);
    }

    if (!latestQr) {
        return res.send(`
            <html>
                <body style="font-family:sans-serif; text-align:center; padding:50px; background:#121212; color:#fff;">
                    <h2>⌛ Generating QR Code...</h2>
                    <p>Please refresh this page in a few seconds.</p>
                    <script>setTimeout(() => location.reload(), 3000);</script>
                </body>
            </html>
        `);
    }

    const qrImageUrl = `https://api.qrserver.com/v1/create-qr-code/?size=300x300&data=${encodeURIComponent(latestQr)}`;
    res.send(`
        <html>
            <body style="font-family:sans-serif; text-align:center; padding:40px; background:#121212; color:#fff;">
                <h1 style="color:#39ff14;">UAV WhatsApp Alert Bot</h1>
                <h3>Scan this QR Code with WhatsApp:</h3>
                <div style="background:#fff; display:inline-block; padding:15px; border-radius:10px;">
                    <img src="${qrImageUrl}" alt="WhatsApp QR Code" width="300" height="300" />
                </div>
                <p style="color:#aaa;">Open WhatsApp on phone → Settings / Menu → Linked Devices → Link a Device</p>
                <script>setTimeout(() => location.reload(), 10000);</script>
            </body>
        </html>
    `);
});

// GET / (Home)
app.get('/', (req, res) => {
    res.redirect('/qr');
});

// Initialize client on server start
console.log('[WHATSAPP BOT] Starting WhatsApp client...');
client.initialize();

// Global Error Catching
process.on('unhandledRejection', (reason, promise) => {
    console.error('[WHATSAPP BOT] Unhandled Rejection:', reason);
});

process.on('uncaughtException', (err) => {
    console.error('[WHATSAPP BOT] Uncaught Exception:', err);
});

// Start HTTP Server
app.listen(PORT, () => {
    console.log(`[WHATSAPP BOT] API listening on http://localhost:${PORT}`);
    console.log(`[WHATSAPP BOT] Web QR Dashboard available at http://localhost:${PORT}/qr`);
});
