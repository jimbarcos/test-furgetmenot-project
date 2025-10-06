<?php
function generate_uuid(){
    // Simple UUID v4
    $data = random_bytes(16);
    $data[6] = chr((ord($data[6]) & 0x0f) | 0x40);
    $data[8] = chr((ord($data[8]) & 0x3f) | 0x80);
    return vsprintf('%s%s-%s-%s-%s-%s%s%s', str_split(bin2hex($data), 4));
}
$errors = [];$successInfo=null;$rawPetType='Auto-Detect';$finalPetType=null;$savedPath=null;$originalName='';$enablePreprocess=true;
if($_SERVER['REQUEST_METHOD']==='POST'){
    $rawPetType = isset($_POST['pet_type']) ? trim($_POST['pet_type']) : 'Auto-Detect';
    $originalName = isset($_POST['pet_name']) ? trim($_POST['pet_name']) : '';
    // Checkbox: if not present, treat as disabled
    $enablePreprocess = isset($_POST['preprocess']) && ($_POST['preprocess']==='on' || $_POST['preprocess']==='1' || $_POST['preprocess']==='true');
    if($originalName===''){ $errors[]='Pet/File Name is required.'; }
    if(!isset($_FILES['pet_image']) || $_FILES['pet_image']['error']!==UPLOAD_ERR_OK){ $errors[]='Pet image is required.'; }
    if(!$errors){
        $uploadTmp = $_FILES['pet_image']['tmp_name'];
        $mime = mime_content_type($uploadTmp);
        if(!preg_match('/image\/(jpeg|png|jpg|webp|bmp)/i',$mime)){
            $errors[]='Unsupported image format.';
        } else {
            // Branch: advanced preprocessing (YOLO) or simple resize only
            if($enablePreprocess){
                $python = 'python';
                $processScript = __DIR__ . DIRECTORY_SEPARATOR . 'python' . DIRECTORY_SEPARATOR . 'process_image.py';
                if(!file_exists($processScript)){
                    $errors[]='Processing script missing.';
                } else {
                    $tmpDir = sys_get_temp_dir();
                    $ext = pathinfo($_FILES['pet_image']['name'], PATHINFO_EXTENSION);
                    if(!$ext) $ext='jpg';
                    $tempInput = $tmpDir.DIRECTORY_SEPARATOR.'reg_'.generate_uuid().'.'.$ext;
                    @move_uploaded_file($uploadTmp,$tempInput);
                    $cmd = escapeshellcmd($python.' '.escapeshellarg($processScript).' '.escapeshellarg($tempInput));
                    $output = shell_exec($cmd.' 2>&1');
                    $json=null; if($output){ $brace=strpos($output,'{'); if($brace!==false){ $json=json_decode(substr($output,$brace),true);} }
                    if(!($json && isset($json['ok']) && $json['ok'])){ $errors[]='Failed to preprocess image.'; }
                    else {
                        $processedBase64 = $json['processed_base64'] ?? null;
                        $detType = $json['pet_type'] ?? 'Unknown';
                        if(!$processedBase64){ $errors[]='Preprocessed image missing.'; }
                        else {
                            if($rawPetType==='Auto-Detect'){ $finalPetType=$detType; }
                            else { $finalPetType=$rawPetType; }
                            // Normalize
                            $finalPetType = strtolower($finalPetType);
                            $folder = null;
                            if(stripos($finalPetType,'cat')===0) $folder='Cats';
                            elseif(stripos($finalPetType,'dog')===0) $folder='Dogs';
                            else { $folder='Unknown'; }
                            if(!$errors){
                                $preDir = __DIR__.DIRECTORY_SEPARATOR.'Preprocessed';
                                $saveDir = $preDir.DIRECTORY_SEPARATOR.$folder;
                                if(!is_dir($saveDir)){ @mkdir($saveDir,0777,true); }
                                $safeBase = preg_replace('/[^a-zA-Z0-9_-]+/','_', $originalName);
                                $uuid = generate_uuid();
                                $fileName = $safeBase.'_'.$uuid.'.jpg';
                                $raw = base64_decode($processedBase64);
                                if($raw===false){ $errors[]='Failed decoding processed image.'; }
                                else {
                                    $dest = $saveDir.DIRECTORY_SEPARATOR.$fileName;
                                    file_put_contents($dest,$raw);
                                    if(file_exists($dest)){
                                        $savedPath = $dest;
                                        $dataUri = 'data:image/jpeg;base64,'.base64_encode($raw);
                                        $successInfo = [
                                            'name'=>$originalName,
                                            'finalType'=> (stripos($finalPetType,'cat')===0?'Cat':(stripos($finalPetType,'dog')===0?'Dog':'Unknown')),
                                            'location'=> 'Preprocessed'.DIRECTORY_SEPARATOR.$folder.DIRECTORY_SEPARATOR.$fileName,
                                            'processedDataUri' => $dataUri
                                        ];
                                    } else { $errors[]='Failed writing file.'; }
                                }
                            }
                        }
                    }
                    @unlink($tempInput);
                }
            } else {
                // Simple resize path: 224x224 without YOLO detection, use Pillow script (no PHP GD dependency)
                $python = 'python';
                $resizeScript = __DIR__ . DIRECTORY_SEPARATOR . 'python' . DIRECTORY_SEPARATOR . 'resize_image.py';
                if(!file_exists($resizeScript)){
                    $errors[]='Resize script missing.';
                } else {
                    $tmpDir = sys_get_temp_dir();
                    $ext = pathinfo($_FILES['pet_image']['name'], PATHINFO_EXTENSION);
                    if(!$ext) $ext='jpg';
                    $tempInput = $tmpDir.DIRECTORY_SEPARATOR.'reg_rs_'.generate_uuid().'.'.$ext;
                    @move_uploaded_file($uploadTmp,$tempInput);
                    $cmd = escapeshellcmd($python.' '.escapeshellarg($resizeScript).' '.escapeshellarg($tempInput));
                    $output = shell_exec($cmd.' 2>&1');
                    $json=null; if($output){ $brace=strpos($output,'{'); if($brace!==false){ $json=json_decode(substr($output,$brace),true);} }
                    if(!($json && isset($json['ok']) && $json['ok'])){ $errors[]='Failed to resize image.'; }
                    else {
                        $processedBase64 = $json['processed_base64'] ?? null;
                        if(!$processedBase64){ $errors[]='Resized image missing.'; }
                        else {
                            // Determine final type from user selection (no auto-detect here)
                            if($rawPetType==='Cat') $finalPetType='Cat';
                            elseif($rawPetType==='Dog') $finalPetType='Dog';
                            else $finalPetType='Unknown';
                            $folder = ($finalPetType==='Cat')?'Cats':(($finalPetType==='Dog')?'Dogs':'Unknown');
                            $preDir = __DIR__.DIRECTORY_SEPARATOR.'Preprocessed';
                            $saveDir = $preDir.DIRECTORY_SEPARATOR.$folder;
                            if(!is_dir($saveDir)){ @mkdir($saveDir,0777,true); }
                            $safeBase = preg_replace('/[^a-zA-Z0-9_-]+/','_', $originalName);
                            $uuid = generate_uuid();
                            $fileName = $safeBase.'_'.$uuid.'.jpg';
                            $raw = base64_decode($processedBase64);
                            if($raw===false){ $errors[]='Failed decoding resized image.'; }
                            else {
                                $dest = $saveDir.DIRECTORY_SEPARATOR.$fileName;
                                file_put_contents($dest,$raw);
                                if(file_exists($dest)){
                                    $savedPath = $dest;
                                    $dataUri = 'data:image/jpeg;base64,'.base64_encode($raw);
                                    $successInfo = [
                                        'name'=>$originalName,
                                        'finalType'=> $finalPetType,
                                        'location'=> 'Preprocessed'.DIRECTORY_SEPARATOR.$folder.DIRECTORY_SEPARATOR.$fileName,
                                        'processedDataUri' => $dataUri
                                    ];
                                } else { $errors[]='Failed writing resized file.'; }
                            }
                        }
                    }
                    @unlink($tempInput);
                }
            }
        }
    }
}
?>
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8" />
<meta name="viewport" content="width=device-width,initial-scale=1.0" />
<title>Register a Missing Pet - Fur-Get Me Not</title>
<link rel="stylesheet" href="css/styles.css" />
<link rel="stylesheet" href="css/components.css" />
<style>
    body { background:#f5f8ff; font-family: 'Segoe UI', Arial, sans-serif; }
    .register-wrapper { max-width:880px; margin:46px auto 80px; padding:26px; background:#fff; border-radius:18px; box-shadow:0 12px 48px -18px rgba(34,58,123,0.18),0 4px 16px -4px rgba(34,58,123,0.12); position:relative; }
    .form-row { display:flex; flex-wrap:wrap; gap:34px; margin-bottom:22px; }
    .form-col { flex:1 1 300px; display:flex; flex-direction:column; gap:8px; }
    label { font-size:1rem; font-weight:600; letter-spacing:.5px; color:#223a7b; margin-bottom:12px; }
    .required { color:#d23b3b; margin-left:4px; }
    input[type=text], select { border:1px solid #c9d5ea; padding:10px 12px; border-radius:10px; font-size:0.92rem; color:#223a7b; background:#f9fbfe; outline:none; transition:border-color .25s, background .25s; }
    input[type=text]:focus, select:focus { border-color:#3867d6; background:#ffffff; }
    .upload-area { border:2px dashed #9fb4d5; background:#f6f9fe; border-radius:14px; padding:38px 18px; text-align:center; font-size:0.95rem; color:#4a5a7b; cursor:pointer; position:relative; transition:border-color .25s, background .25s, color .25s; }
    .upload-area.dragover { border-color:#3867d6; background:#eef4ff; color:#223a7b; }
    .btn-row { display:flex; gap:18px; margin-top:26px; }
    .btn-primary { width: 100%; background:#3867d6; color:#fff; font-weight:600; border:none; border-radius:24px; padding:12px; font-size:1.1rem; cursor:pointer; transition:background .3s, ease; }
    .btn-primary:hover { background:#2d4ba0; }
    .btn-secondary { width: 100%; background:#ffd166; color:#fff; font-weight:600; border:none; border-radius:24px; padding:12px; font-size:1.1rem; cursor:pointer; transition:filter .3s, ease; }
    .btn-secondary:hover { background:#f6c94c; }
    .error-box { background:#ffe9e9; border:1px solid #efb1b1; color:#7d1f1f; padding:14px 18px; border-radius:12px; font-size:0.85rem; margin-bottom:18px; }
    .success-modal-backdrop { position:fixed; inset:0; background:rgba(10,20,40,.55); backdrop-filter:blur(3px); display:flex; align-items:center; justify-content:center; z-index:1000; }
    .success-modal { background:#fff; padding:34px 40px 42px; border-radius:28px; max-width:520px; width:92%; box-shadow:0 18px 48px -12px rgba(0,0,0,.32); display:flex; flex-direction:column; gap:18px; }
    .success-modal h2 { margin:0; font-size:1.6rem; color:#223a7b; text-align:center; }
    .success-details { font-size:0.9rem; color:#223a7b; background:#f0f5ff; border:1px solid #d4e1f5; padding:14px 16px; border-radius:14px; }
    .close-success { align-self:center; background:#3867d6; color:#fff; font-weight:700; border:none; border-radius:14px; padding:10px 28px; cursor:pointer; font-size:0.9rem; }
    .preview-img { max-width:220px; max-height:220px; border-radius:16px; object-fit:cover; box-shadow:0 6px 20px -8px rgba(34,58,123,.4); display:none; margin-top:18px; }
    .reg-image-placeholder-text { color: #3867d6; font-size: 1.1rem; font-weight: 600; margin-bottom: 8px; padding: 0 12px;}

    .reg-remove-image {
        position: absolute;
        top: -12px;
        right: -12px;
        width: 28px;
        height: 28px;
        background: #e74c3c;
        border: none;
        border-radius: 50%;
        color: white;
        font-size: 20px;
        display: flex;
        align-items: center;
        justify-content: center;
        cursor: pointer;
        box-shadow: 0 2px 8px rgba(0,0,0,0.2);
        transition: all 0.2s ease;
        z-index: 2;
    }

    .reg-remove-image:hover {
        background: #c0392b;
        transform: scale(1.1);
    }

    @media (max-width: 768px) {
        .reg-image-placeholder-text { font-size: 1rem; text-align: center; }
        .btn-row { flex-direction: column; gap: 12px; }
        .reg-remove-image {
            width: 32px;
            height: 32px;
            top: -16px;
            right: -16px;
            font-size: 22px;
        }
    }
</style>
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
    <header>
        <div class="header-bar"></div>
        <a href="index.php" class="back-arrow">
            <img src="assets/How-it-Works/back-arrow.png" alt="Back">
        </a>
        <div class="subpage-header">
            <img src="assets/Logos/interface-setting-app-widget--Streamline-Core.png" alt="Matches Icon" class="subpage-icon">
            <h1 class="subpage-title">Register a missing pet</h1>
            <div class="subpage-subpill">Upload your missing pet</div>
        </div>
    </header>
<main>
    <!-- Notification dialog -->
    <div id="register-notification-dialog" class="find-notification-dialog" style="display:none;">
        <span id="register-notification-dialog-text"></span>
    </div>
    <!-- Loading overlay -->
    <div id="register-loading-overlay" style="display:none;position:fixed;top:0;left:0;width:100vw;height:100vh;z-index:9999;background:rgba(248,250,255,0.85);backdrop-filter:blur(2px);align-items:center;justify-content:center;flex-direction:column;">
        <div style="display:flex;flex-direction:column;align-items:center;">
            <div class="find-spinner" style="width:64px;height:64px;margin-bottom:18px;">
                <svg viewBox="0 0 50 50" style="width:100%;height:100%;">
                    <circle cx="25" cy="25" r="20" stroke="#3867d6" stroke-width="6" fill="none" stroke-linecap="round" stroke-dasharray="31.4 31.4">
                        <animateTransform attributeName="transform" type="rotate" repeatCount="indefinite" dur="1s" values="0 25 25;360 25 25"/>
                    </circle>
                </svg>
            </div>
            <div id="register-overlay-title" style="font-size:1.25rem;color:#223a7b;font-weight:700;margin-bottom:8px;">Processing your image...</div>
            <div id="register-overlay-desc" style="font-size:1.05rem;color:#3867d6;text-align:center;">This may take a few seconds.<br>Please wait while we detect and crop your pet's face.</div>
        </div>
    </div>
    <form class="register-wrapper" id="registerForm" method="post" enctype="multipart/form-data">
        <?php if($errors){ echo '<div class="error-box"><b>Please correct the following:</b><br>'.implode('<br>',$errors).'</div>'; } ?>
        <div class="form-row">
            <div class="form-col">
                <label>Pet/File Name <span class="required">*</span></label>
                <input type="text" name="pet_name" value="<?php echo htmlspecialchars($originalName); ?>" placeholder="e.g. Luna (Sphinx)"/>
            </div>
            <div class="form-col">
                <label>Pet Type <span class="required">*</span></label>
                <select name="pet_type">
                    <?php
                        $opts=['Auto-Detect','Cat','Dog'];
                        foreach($opts as $o){ $sel = ($rawPetType===$o)?'selected':''; echo '<option '.$sel.'>'.htmlspecialchars($o).'</option>'; }
                    ?>
                </select>
            </div>
        </div>
        <div class="form-row" style="margin-top:6px;">
            <div class="form-col" style="flex:1 1 100%;">
                <label>Upload Pet Image: <span class="required">*</span></label>
                <div class="find-image-upload" id="reg-upload-box" style="min-height:240px;position:relative;">
                    <input type="file" id="reg-pet-image" name="pet_image" accept="image/*" class="find-input" style="display:none;">
                    <label for="reg-pet-image" id="reg-upload-label" class="find-image-upload-label" style="cursor:pointer;min-height:240px;display:flex;flex-direction:column;align-items:center;justify-content:center;">
                        <span id="reg-image-placeholder" class="reg-image-placeholder-text">Click to upload or drag an image here</span>
                        <img id="reg-image-preview" src="" alt="Preview" style="display:none;max-width:100%;max-height:200px;border-radius:14px;box-shadow:0 2px 8px rgba(60,90,200,0.10);margin-top:12px;object-fit:cover;" />
                    </label>
                    <button type="button" id="reg-remove-image" class="reg-remove-image" style="display:none;" aria-label="Remove uploaded image">×</button>
                </div>
            </div>
        </div>
        <!-- Image Pre-processing option -->
        <div class="find-form-group">
            <label class="find-section-title" for="preprocess" style="margin-bottom:10px;">Image Pre-processing</label>
            <div class="find-checkbox-group">
                <input type="checkbox" id="preprocess" name="preprocess" checked style="width:18px;height:18px;">
                <label for="preprocess">Enable advanced image pre-processing</label>
            </div>
            <div class="find-form-description">Image pre-processing improves detection accuracy by normalizing images into a standard body/face crop of pet and resizing it to 224x224. Disable only if you experience issues.</div>
        </div>
        <div class="btn-row">
            <button type="submit" class="btn-primary">Register missing pet</button>
            <button type="button" class="btn-secondary" onclick="window.location.href='index.php'">Cancel</button>
        </div>
    </form>
</main>
<?php if($successInfo){ ?>
<div class="success-modal-backdrop" id="successModal">
    <div class="success-modal" role="dialog" aria-modal="true">
        <h2>Pet Registered!</h2>
        <div class="success-details">
            <div><b>Pet/File Name:</b> <?php echo htmlspecialchars($successInfo['name']); ?></div>
            <div><b>Pet Type:</b> <?php echo htmlspecialchars($successInfo['finalType']); ?></div>
            <div><b>Saved Location:</b> <?php echo htmlspecialchars($successInfo['location']); ?></div>
        </div>
        <?php if(!empty($successInfo['processedDataUri'])){ ?>
            <div style="text-align:center;">
                <div style="font-size:0.8rem;letter-spacing:.5px;text-transform:uppercase;font-weight:600;color:#223a7b;margin-bottom:8px;">Preprocessed Image</div>
                <img src="<?php echo $successInfo['processedDataUri']; ?>" alt="Preprocessed" style="max-width:260px;max-height:260px;border-radius:18px;box-shadow:0 6px 22px -10px rgba(34,58,123,0.35);object-fit:cover;" />
            </div>
        <?php } ?>
        <button class="close-success" onclick="document.getElementById('successModal').remove();">Close</button>
    </div>
</div>
<?php } ?>
<script>
// Registration image upload logic (mirrors find page style, with taller box)
const regInput = document.getElementById('reg-pet-image');
const regPreview = document.getElementById('reg-image-preview');
const regPlaceholder = document.getElementById('reg-image-placeholder');
const regBox = document.getElementById('reg-upload-box');
const regRemoveBtn = document.getElementById('reg-remove-image');

function resetRegImageUpload() {
    regInput.value = ''; // Clear the file input
    regPreview.src = '';
    regPreview.style.display = 'none';
    regPlaceholder.style.display = 'block';
    regRemoveBtn.style.display = 'none';
}

// Handle label click to prevent double dialog when image is uploaded
const regUploadLabel = document.getElementById('reg-upload-label');
regUploadLabel.addEventListener('click', function(e) {
    if (hasRegisterImage()) {
        e.preventDefault(); // Prevent file dialog if image already uploaded
    }
});

regInput.addEventListener('change', e => {
    const file = e.target.files[0];
    if (file && file.type.startsWith('image/')) {
        const reader = new FileReader();
        reader.onload = ev => { 
            regPreview.src = ev.target.result; 
            regPreview.style.display='block'; 
            regPlaceholder.style.display='none'; 
            regRemoveBtn.style.display='flex';
        };
        reader.readAsDataURL(file);
    } else {
        resetRegImageUpload();
    }
});
// Add remove image functionality
regRemoveBtn.addEventListener('click', function(e) {
    e.stopPropagation(); // Prevent triggering upload area click
    resetRegImageUpload();
});

// Drag & drop
['dragenter','dragover'].forEach(ev=>regBox.addEventListener(ev, e=>{e.preventDefault();regBox.style.borderColor='#3867d6';}));
['dragleave','drop'].forEach(ev=>regBox.addEventListener(ev, e=>{e.preventDefault();regBox.style.borderColor='#b3c6ff';}));
regBox.addEventListener('drop', e=>{ 
    const files = e.dataTransfer.files; 
    if(files.length && files[0].type.startsWith('image/')){ 
        regInput.files=files; 
        regInput.dispatchEvent(new Event('change')); 
    } else {
        resetRegImageUpload();
    }
});

// Notification dialog & validation logic
const registerDialog = document.getElementById('register-notification-dialog');
const registerDialogText = document.getElementById('register-notification-dialog-text');
const registerLoading = document.getElementById('register-loading-overlay');
const registerForm = document.getElementById('registerForm');
let registerDialogTimeout = null;
const preprocessCheckbox = document.getElementById('preprocess');
const overlayTitle = document.getElementById('register-overlay-title');
const overlayDesc = document.getElementById('register-overlay-desc');

function showRegisterDialog(message){
    registerDialogText.textContent = message;
    registerDialog.classList.remove('hide');
    registerDialog.classList.add('show');
    registerDialog.style.display='flex';
    if(registerDialogTimeout) clearTimeout(registerDialogTimeout);
    registerDialogTimeout = setTimeout(hideRegisterDialog, 2500);
}
function hideRegisterDialog(){
    registerDialog.classList.remove('show');
    registerDialog.classList.add('hide');
    setTimeout(()=>{ registerDialog.style.display='none'; registerDialogText.textContent=''; },350);
}
function hasRegisterImage(){
    return regInput.files && regInput.files.length>0 && regInput.files[0].type.startsWith('image/');
}

registerForm.addEventListener('submit', e=>{
    const nameField = registerForm.querySelector('input[name="pet_name"]');
    if(!nameField.value.trim()){
        e.preventDefault();
        showRegisterDialog('Please enter a pet/file name.');
        nameField.focus();
        return;
    }
    if(!hasRegisterImage()){
        e.preventDefault();
        showRegisterDialog('Please upload a pet image first.');
        return;
    }
    // Show loading overlay while backend preprocesses
    if(preprocessCheckbox && !preprocessCheckbox.checked){
        if(overlayTitle) overlayTitle.textContent='Resizing your image...';
        if(overlayDesc) overlayDesc.textContent='Quickly resizing to 224×224 (no detection).';
    }else{
        if(overlayTitle) overlayTitle.textContent='Processing your image...';
        if(overlayDesc) overlayDesc.innerHTML='This may take a few seconds.<br>Please wait while we detect and crop your pet\'s face.';
    }
    registerLoading.style.display='flex';
});

// Success modal redirect (if present)
const successModal = document.getElementById('successModal');
if(successModal){
    const closeBtn = successModal.querySelector('.close-success');
    if(closeBtn){
        closeBtn.addEventListener('click', ()=>{
            successModal.remove();
            window.location.href='register-missing-pet.php';
        });
    }
}
</script>
<script src="js/page-transitions.js"></script>
</body>
</html>
