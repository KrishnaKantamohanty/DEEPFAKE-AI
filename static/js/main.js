document.addEventListener('DOMContentLoaded', () => {
    // --- SPA ROUTING ---
    const navItems = document.querySelectorAll('.nav-item');
    const pageSections = document.querySelectorAll('.page-section');

    navItems.forEach(item => {
        item.addEventListener('click', () => {
            // Remove active classes
            navItems.forEach(nav => nav.classList.remove('active'));
            pageSections.forEach(sec => sec.classList.remove('active'));

            // Add active to clicked
            item.classList.add('active');
            const targetId = item.getAttribute('data-target');
            document.getElementById(targetId).classList.add('active');

            // Re-render history if navigating to history
            if(targetId === 'history') {
                renderHistory();
            }
            // Init charts if navigating to analytics and they aren't init yet
            if(targetId === 'analytics') {
                initCharts();
            }
        });
    });

    // --- HEALTH CHECK ---
    const sidebarHealth = document.getElementById('sidebar-health');
    fetch('/health')
        .then(res => res.json())
        .then(data => {
            const statusInd = sidebarHealth.querySelector('.status-indicator');
            const statusVal = sidebarHealth.querySelector('.status-val');
            if(data.status === 'ok') {
                statusVal.innerText = 'Online - Ready';
                statusInd.style.background = 'var(--status-real)';
                statusInd.style.boxShadow = '0 0 10px var(--status-real-glow)';
            } else {
                statusVal.innerText = 'Offline';
                statusInd.style.background = 'var(--status-fake)';
                statusInd.style.boxShadow = '0 0 10px var(--status-fake-glow)';
            }
        }).catch(err => {
            const statusVal = sidebarHealth.querySelector('.status-val');
            statusVal.innerText = 'Connection Error';
        });

    // --- DRAG AND DROP & FILE SELECTION ---
    const dropZone = document.getElementById('drop-zone');
    const fileInput = document.getElementById('file-input');
    const previewContainer = document.getElementById('preview-container');
    const imagePreview = document.getElementById('image-preview');
    const clearBtn = document.getElementById('clear-btn');
    const predictBtn = document.getElementById('predict-btn');
    const resultCard = document.getElementById('result-card');

    let currentFile = null;

    dropZone.addEventListener('dragover', (e) => {
        e.preventDefault();
        dropZone.classList.add('dragover');
    });
    dropZone.addEventListener('dragleave', (e) => {
        e.preventDefault();
        dropZone.classList.remove('dragover');
    });
    dropZone.addEventListener('drop', (e) => {
        e.preventDefault();
        dropZone.classList.remove('dragover');
        if(e.dataTransfer.files.length > 0) {
            handleFile(e.dataTransfer.files[0]);
        }
    });
    fileInput.addEventListener('change', (e) => {
        if(e.target.files.length > 0) {
            handleFile(e.target.files[0]);
        }
    });

    function handleFile(file) {
        if(!file.type.startsWith('image/')) {
            alert('Please select a valid image file.');
            return;
        }
        currentFile = file;
        
        // Setup preview
        const reader = new FileReader();
        reader.onload = (e) => {
            imagePreview.src = e.target.result;
            dropZone.classList.add('hidden');
            previewContainer.classList.remove('hidden');
            predictBtn.disabled = false;
            resultCard.classList.add('hidden'); // Hide previous result
        };
        reader.readAsDataURL(file);
    }

    clearBtn.addEventListener('click', () => {
        currentFile = null;
        fileInput.value = '';
        imagePreview.src = '';
        previewContainer.classList.add('hidden');
        dropZone.classList.remove('hidden');
        predictBtn.disabled = true;
        resultCard.classList.add('hidden');
    });

    // --- PREDICTION LOGIC ---
    const uploadForm = document.getElementById('upload-form');
    const loadingOverlay = document.getElementById('loading-overlay');
    const loadingText = document.getElementById('loading-text');
    const loadingSubtext = document.getElementById('loading-subtext');

    uploadForm.addEventListener('submit', (e) => {
        e.preventDefault();
        if(!currentFile) return;

        // Start multi-step loading animation
        loadingOverlay.classList.remove('hidden');
        
        const loadingSteps = [
            { t: 'Analyzing image...', s: 'Extracting visual features' },
            { t: 'Running inference...', s: 'Passing tensors through Neural Network' },
            { t: 'Evaluating authenticity...', s: 'Calculating confidence probabilities' }
        ];
        
        let step = 0;
        const interval = setInterval(() => {
            step++;
            if(step < loadingSteps.length) {
                loadingText.innerText = loadingSteps[step].t;
                loadingSubtext.innerText = loadingSteps[step].s;
            }
        }, 800);

        const formData = new FormData();
        formData.append('image', currentFile);
        
        const startTime = Date.now();

        fetch('/predict', {
            method: 'POST',
            body: formData
        })
        .then(response => response.json())
        .then(data => {
            clearInterval(interval);
            loadingOverlay.classList.add('hidden');
            
            const latency = Date.now() - startTime;
            
            if(data.error) {
                alert('Error: ' + data.error);
                return;
            }

            displayResult(data, currentFile.name, latency);
            saveToHistory(data, currentFile.name, imagePreview.src, latency);
            updateStats();
        })
        .catch(err => {
            clearInterval(interval);
            loadingOverlay.classList.add('hidden');
            alert('Failed to connect to inference server.');
        });
    });

    function displayResult(data, filename, latency) {
        resultCard.classList.remove('hidden');
        document.getElementById('inference-time').innerText = `Latency: ${latency}ms`;

        const isFake = data.classification.includes('FAKE');
        
        const statusDiv = document.getElementById('result-status');
        const fillBar = document.getElementById('confidence-fill');
        
        if(isFake) {
            statusDiv.innerHTML = `<div class="status-badge is-fake"><i class="fa-solid fa-triangle-exclamation"></i> AI-Generated Media</div>`;
            fillBar.style.width = '100%'; 
            fillBar.style.background = 'var(--status-fake)';
            fillBar.style.boxShadow = '0 0 15px var(--status-fake-glow)';
        } else {
            statusDiv.innerHTML = `<div class="status-badge is-real"><i class="fa-solid fa-circle-check"></i> Authentic Media</div>`;
            fillBar.style.width = '0%'; // visually implies authentic is 0, fake is 100 on the slider
            fillBar.style.background = 'var(--status-real)';
            fillBar.style.boxShadow = '0 0 15px var(--status-real-glow)';
        }

        document.getElementById('prob-real').innerText = data.real_probability.toFixed(1) + '%';
        document.getElementById('prob-fake').innerText = data.fake_probability.toFixed(1) + '%';
        
        // Animate fill bar (width based on fake probability)
        setTimeout(() => {
            fillBar.style.width = data.fake_probability.toFixed(0) + '%';
        }, 100);
    }

    // --- LOCAL STORAGE HISTORY ---
    function saveToHistory(data, filename, imageDataUrl, latency) {
        let history = JSON.parse(localStorage.getItem('nexus_history') || '[]');
        const isFake = data.classification.includes('FAKE');
        const entry = {
            id: Date.now(),
            time: new Date().toLocaleString(),
            name: filename,
            class: isFake ? 'AI-Generated' : 'Authentic',
            conf: data.confidence.toFixed(1) + '%',
            isFake: isFake,
            image: imageDataUrl,
            data: data,
            latency: latency
        };
        history.unshift(entry); // add to front
        if(history.length > 30) history.pop(); // keep last 30
        
        // Save to localStorage with Quota error handling
        try {
            localStorage.setItem('nexus_history', JSON.stringify(history));
        } catch (e) {
            console.warn("Storage quota reached. Clearing older image payloads.");
            // Clear image data from older history entries to free space
            for (let i = history.length - 1; i >= 0; i--) {
                if (history[i].image) {
                    history[i].image = null;
                    try {
                        localStorage.setItem('nexus_history', JSON.stringify(history));
                        break;
                    } catch (err) {}
                }
            }
        }
    }

    function renderHistory() {
        const tbody = document.getElementById('history-body');
        let history = JSON.parse(localStorage.getItem('nexus_history') || '[]');
        
        if(history.length === 0) {
            tbody.innerHTML = '<tr class="empty-row"><td colspan="5">No history found for this session.</td></tr>';
            return;
        }

        tbody.innerHTML = '';
        history.forEach(item => {
            const tr = document.createElement('tr');
            const pillClass = item.isFake ? 'pill-fake' : 'pill-real';
            tr.innerHTML = `
                <td>${item.time}</td>
                <td><strong>${item.name}</strong></td>
                <td><span class="status-pill ${pillClass}">${item.class}</span></td>
                <td>${item.conf}</td>
                <td><button class="btn-secondary btn-view-history" data-id="${item.id}" style="padding: 4px 8px; font-size: 0.75rem;">View</button></td>
            `;
            tbody.appendChild(tr);
        });
    }

    // Set up click delegator on history table
    const historyTable = document.getElementById('history-table');
    if (historyTable) {
        historyTable.addEventListener('click', (e) => {
            if (e.target.classList.contains('btn-view-history')) {
                const itemId = e.target.getAttribute('data-id');
                viewHistoryItem(itemId);
            }
        });
    }

    function viewHistoryItem(id) {
        let history = JSON.parse(localStorage.getItem('nexus_history') || '[]');
        const item = history.find(x => x.id == id);
        if(!item) return;

        // Switch section to Dashboard
        navItems.forEach(nav => nav.classList.remove('active'));
        pageSections.forEach(sec => sec.classList.remove('active'));
        
        const dashboardNav = Array.from(navItems).find(nav => nav.getAttribute('data-target') === 'dashboard');
        if(dashboardNav) dashboardNav.classList.add('active');
        const dashboardSection = document.getElementById('dashboard');
        if(dashboardSection) dashboardSection.classList.add('active');

        // Set image preview
        if(item.image) {
            imagePreview.src = item.image;
            dropZone.classList.add('hidden');
            previewContainer.classList.remove('hidden');
            predictBtn.disabled = false;
        } else {
            // If image was purged to save space
            imagePreview.src = '';
            dropZone.classList.remove('hidden');
            previewContainer.classList.add('hidden');
            predictBtn.disabled = true;
        }

        // Show result details
        displayResult(item.data, item.name, item.latency || 0);
        
        // Smooth scroll to top of main content
        window.scrollTo({ top: 0, behavior: 'smooth' });
    }

    // Clear History Button Click Listener
    const clearHistoryBtn = document.getElementById('clear-history-btn');
    if (clearHistoryBtn) {
        clearHistoryBtn.addEventListener('click', () => {
            if (confirm('Are you sure you want to clear all prediction history?')) {
                localStorage.removeItem('nexus_history');
                renderHistory();
                updateStats();
            }
        });
    }

    // Update Dashboard stats based on history
    function updateStats() {
        let history = JSON.parse(localStorage.getItem('nexus_history') || '[]');
        let baseScanned = 1482; // mocked base
        let baseFakes = 439;
        
        let localFakes = history.filter(h => h.isFake).length;
        
        document.getElementById('stat-scanned').innerText = (baseScanned + history.length).toLocaleString();
        document.getElementById('stat-fakes').innerText = (baseFakes + localFakes).toLocaleString();
    }
    updateStats(); // run on load

    // --- CHARTS (Mock Analytics) ---
    let chartsInit = false;
    function initCharts() {
        if(chartsInit) return;
        chartsInit = true;

        // Line Chart
        const ctxLine = document.getElementById('lineChart').getContext('2d');
        new Chart(ctxLine, {
            type: 'line',
            data: {
                labels: ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun'],
                datasets: [{
                    label: 'Media Scans',
                    data: [120, 190, 150, 220, 180, 250, 310],
                    borderColor: '#6366F1',
                    backgroundColor: 'rgba(99, 102, 241, 0.1)',
                    tension: 0.4,
                    fill: true
                }]
            },
            options: {
                responsive: true,
                plugins: { legend: { display: false } },
                scales: {
                    y: { grid: { color: 'rgba(255,255,255,0.05)' }, ticks: { color: '#94A3B8' } },
                    x: { grid: { display: false }, ticks: { color: '#94A3B8' } }
                }
            }
        });

        // Doughnut Chart
        const ctxDoughnut = document.getElementById('doughnutChart').getContext('2d');
        new Chart(ctxDoughnut, {
            type: 'doughnut',
            data: {
                labels: ['Authentic', 'AI-Generated'],
                datasets: [{
                    data: [70, 30],
                    backgroundColor: ['#10B981', '#EF4444'],
                    borderWidth: 0
                }]
            },
            options: {
                responsive: true,
                cutout: '75%',
                plugins: {
                    legend: { position: 'bottom', labels: { color: '#fff' } }
                }
            }
        });
    }
});
