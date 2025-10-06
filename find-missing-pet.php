<!DOCTYPE html>
<html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Find Missing Pet - Fur-Get Me Not</title>
        <link rel="stylesheet" href="css/styles.css">
    </head>

    <body>
        <div id="page-transition-overlay">
            <div class="loading-content">
                <div class="loading-paw">
                    <img src="assets/Logos/pawprint-blue 1.png" alt="Loading">
                </div>
                <p class="loading-text">Loading...</p>
            </div>
        </div>
        <div class="bg-paws"></div>

        <!-- Loading overlay -->
        <div id="find-loading-overlay" style="display:none;position:fixed;top:0;left:0;width:100vw;height:100vh;z-index:9999;background:rgba(248,250,255,0.85);backdrop-filter:blur(2px);align-items:center;justify-content:center;flex-direction:column;">
            <div style="display:flex;flex-direction:column;align-items:center;max-width:520px;">
                <div class="find-spinner" style="width:64px;height:64px;margin-bottom:18px;">
                    <svg viewBox="0 0 50 50" style="width:100%;height:100%;">
                        <circle cx="25" cy="25" r="20" fill="none" stroke="#3867d6" stroke-width="6" stroke-linecap="round" stroke-dasharray="90 60" stroke-dashoffset="0">
                            <animateTransform attributeName="transform" type="rotate" from="0 25 25" to="360 25 25" dur="1s" repeatCount="indefinite"/>
                        </circle>
                    </svg>
                </div>
                <div id="find-progress-stage" role="status" aria-live="polite" style="font-size:1.1rem;color:#223a7b;font-weight:700;margin-bottom:14px;text-align:center;min-height:26px;display:flex;align-items:center;justify-content:center;">Starting...</div>
                <div id="find-progress-detail" aria-live="polite" style="min-height:48px;display:flex;align-items:center;justify-content:center;padding:0 6px;font-size:0.9rem;color:#3867d6;text-align:center;line-height:1.35;">Preparing preprocessing pipeline...</div>
            </div>
        </div>

        <header>
            <div class="header-bar"></div>
            <a href="index.php" class="back-arrow">
                <img src="assets/How-it-Works/back-arrow.png" alt="Back">
            </a>
            
            <div class="container">
                <div class="subpage-header">
                    <img src="assets/Logos/pawprint-blue 1.png" alt="Pawprint Logo" class="subpage-icon">
                    <h1 class="subpage-title">Find a missing pet</h1>
                    <div class="subpage-subpill">Upload a lost pet</div>
                </div>
            </div>
        </header>

        <main class="find-main">
            <!-- Notification Dialog -->
            <div id="find-notification-dialog" class="find-notification-dialog" style="display:none;">
                <span id="find-notification-dialog-text"></span>
            </div>

            <!-- Form Card -->
            <div class="find-form-card">
                <form id="findForm" action="see-matches.php" method="post" enctype="multipart/form-data">
                    
                    <input type="hidden" name="client_start_ms" id="client_start_ms" value="">
                    
                    <!-- Image Upload Section -->
                    <div class="find-form-group">
                        <label for="pet-image">Upload Pet Image: <span class="find-required">*</span></label>
                        
                        <div class="find-upload-area" id="upload-area">
                            <input type="file" id="pet-image" name="pet-image" accept="image/*" style="display:none;">
                            <div class="find-upload-content">
                                <div class="find-upload-text" id="upload-text">Click to upload or drag an image here</div>
                                <div class="find-upload-hint">JPG or PNG files accepted</div>
                            </div>
                            <img id="image-preview" class="find-image-preview" src="" alt="Preview" style="display:none;">
                            <button type="button" id="remove-image" class="find-remove-image" style="display:none;" aria-label="Remove uploaded image">×</button>
                        </div>
                    </div>

                    <!-- Pet Type Section -->
                    <div class="find-form-group">
                        <label for="pet-type">Pet Type:</label>
                        <select id="pet-type" name="pet-type" class="find-pet-type">
                            <option value="">Auto-Detect</option>
                            <option value="Dog">Dog</option>
                            <option value="Cat">Cat</option>
                        </select>
                        <div class="find-form-hint">Filter results by pet type</div>
                    </div>

                    <!-- Image Pre-processing Section -->
                    <div class="find-form-group">
                        <div class="find-section-title">Image Pre-processing</div>
                        
                        <div class="find-checkbox-group">
                            <input type="checkbox" id="preprocess" name="preprocess" checked>
                            <label for="preprocess">Enable advanced image pre-processing</label>
                        </div>

                        <div class="find-form-description">
                            Image pre-processing improves detection accuracy by normalizing images into a standard body/face crop of pet and resizing it to 224x224.
                            <br><br>
                            <strong>Disable only if you experience issues.</strong>
                        </div>
                    </div>

                    <!-- Submit Button -->
                    <button type="submit" class="find-submit-btn">Submit</button>
                </form>
            </div>
            
            <script>
            // Image upload preview logic
            const imageInput = document.getElementById('pet-image');
            const imagePreview = document.getElementById('image-preview');
            const uploadText = document.getElementById('upload-text');
            const uploadArea = document.getElementById('upload-area');
            const findForm = document.getElementById('findForm');
            
            // Dialog elements
            const findNotificationDialog = document.getElementById('find-notification-dialog');
            const findNotificationDialogText = document.getElementById('find-notification-dialog-text');
            const findLoadingOverlay = document.getElementById('find-loading-overlay');
            let dialogTimeout = null;

            function showDialog(message) {
                findNotificationDialogText.textContent = message;
                findNotificationDialog.classList.remove('hide');
                findNotificationDialog.classList.add('show');
                findNotificationDialog.style.display = 'flex';
                if (dialogTimeout) clearTimeout(dialogTimeout);
                dialogTimeout = setTimeout(hideDialog, 2500);
            }
            function hideDialog() {
                findNotificationDialog.classList.remove('show');
                findNotificationDialog.classList.add('hide');
                setTimeout(() => {
                    findNotificationDialog.style.display = 'none';
                    findNotificationDialogText.textContent = '';
                }, 350);
            }

            function hasImageUploaded() {
                return imageInput.files && imageInput.files.length > 0 && imageInput.files[0].type.startsWith('image/');
            }

            // Click to upload
            uploadArea.addEventListener('click', function() {
                if (!hasImageUploaded()) {
                    imageInput.click();
                }
            });

            imageInput.addEventListener('change', function(e) {
                const file = e.target.files[0];
                if (file && file.type.startsWith('image/')) {
                    const reader = new FileReader();
                    reader.onload = function(ev) {
                        imagePreview.src = ev.target.result;
                        imagePreview.style.display = 'block';
                        uploadText.style.display = 'none';
                        uploadArea.querySelector('.find-upload-hint').style.display = 'none';
                        removeImageBtn.style.display = 'flex';
                    };
                    reader.readAsDataURL(file);
                    hideDialog();
                } else {
                    resetImageUpload();
                }
            });

            const removeImageBtn = document.getElementById('remove-image');
            
            function resetImageUpload() {
                imageInput.value = ''; // Clear the file input
                imagePreview.src = '';
                imagePreview.style.display = 'none';
                uploadText.style.display = 'block';
                uploadArea.querySelector('.find-upload-hint').style.display = 'block';
                removeImageBtn.style.display = 'none';
            }

            // Add remove image functionality
            removeImageBtn.addEventListener('click', function(e) {
                e.stopPropagation(); // Prevent triggering upload area click
                resetImageUpload();
            });

            // Drag and drop support
            uploadArea.addEventListener('dragover', function(e) {
                e.preventDefault();
                uploadArea.classList.add('find-dragover');
            });
            uploadArea.addEventListener('dragleave', function(e) {
                e.preventDefault();
                uploadArea.classList.remove('find-dragover');
            });
            uploadArea.addEventListener('drop', function(e) {
                e.preventDefault();
                uploadArea.classList.remove('find-dragover');
                const files = e.dataTransfer.files;
                if (files.length > 0 && files[0].type.startsWith('image/')) {
                    imageInput.files = files;
                    const event = new Event('change');
                    imageInput.dispatchEvent(event);
                } else {
                    imageInput.value = '';
                    resetImageUpload();
                    showDialog('Please upload a valid image file.');
                }
            });

            // Dialog notification on submit if image missing
            findForm.addEventListener('submit', function(e) {
                if (!hasImageUploaded()) {
                    e.preventDefault();
                    showDialog('Please upload a pet image before submitting.');
                } else {
                    hideDialog();
                    // Show loading overlay with staged progress simulation
                    findLoadingOverlay.style.display = 'flex';
                    // Record client start timestamp (ms since epoch)
                    const tsField = document.getElementById('client_start_ms');
                    if(tsField){ tsField.value = Date.now(); }
                    findForm.querySelector('button[type="submit"]').disabled = true;
                    simulateFindProgress();
                }
            });

            // Staged description simulation (client-side only; real server steps happen after submit)
            const progressStage = document.getElementById('find-progress-stage');
            const progressDetail = document.getElementById('find-progress-detail');

            function simulateFindProgress(){
                const stages = [
                    {t:0,    stage:'Initializing', detail:'Preparing preprocessing pipeline...'},
                    {t:3000,  stage:'Reading Image', detail:'Loading file & validating format...'},
                    {t:4000,  stage:'Detecting Pet', detail:'Running YOLO face/body detection...'},
                    {t:5000, stage:'Cropping', detail:'Extracting detected region & cleaning background...'},
                    {t:4000, stage:'Normalizing', detail:'Resizing to 224x224 and enhancing quality...'},
                    {t:6000, stage:'Preparing Matches', detail:'Comparing against stored pet dataset...'},
                    {t:3600, stage:'Finalizing', detail:'Sorting and formatting match results...'}
                ];
                stages.forEach(s=>{
                    setTimeout(()=>{
                        progressStage.textContent = s.stage;
                        progressDetail.textContent = s.detail;
                    }, s.t);
                });
                setTimeout(()=>{
                    progressDetail.textContent = 'Almost done...';
                }, 5000);
            }
            </script>
            <script src="js/page-transitions.js"></script>
        </main>
    </body>
</html>