/**
 * Main UI controller for deepfake detection app
 * Handles permissions, service lifecycle, and file analysis API
 */
package com.example.deepfakeguard

import android.Manifest
import android.content.ComponentName
import android.content.Context
import android.content.Intent
import android.content.ServiceConnection
import android.content.pm.PackageManager
import android.net.Uri
import android.os.Build
import android.os.Bundle
import android.os.IBinder
import android.provider.Settings
import android.view.View
import android.widget.*
import androidx.activity.result.contract.ActivityResultContracts
import androidx.appcompat.app.AlertDialog
import androidx.appcompat.app.AppCompatActivity

import androidx.core.content.ContextCompat
import androidx.lifecycle.lifecycleScope
import kotlinx.coroutines.launch
import timber.log.Timber

class MainActivity : AppCompatActivity(), ServiceConnection {
    
    companion object {
        // Essential permissions for call monitoring and audio analysis
        private val REQUIRED_PERMISSIONS = arrayOf(
            Manifest.permission.RECORD_AUDIO,
            Manifest.permission.READ_PHONE_STATE,
            Manifest.permission.READ_CALL_LOG,
            Manifest.permission.MODIFY_AUDIO_SETTINGS,
            Manifest.permission.FOREGROUND_SERVICE,
            Manifest.permission.FOREGROUND_SERVICE_PHONE_CALL
        )
    }
    
    // Core UI components
    private lateinit var tvStatus: TextView
    private lateinit var tvServiceStatus: TextView
    private lateinit var tvModelStatus: TextView
    private lateinit var btnToggleService: Button
    private lateinit var btnPermissions: Button
    private lateinit var btnSettings: Button
    private lateinit var progressBar: ProgressBar
    private lateinit var switchAutoStart: Switch
    
    // File analysis UI
    private lateinit var btnSelectAudioFile: Button
    private lateinit var btnAnalyzeFile: Button
    private lateinit var tvSelectedFile: TextView
    private lateinit var layoutAnalysisResults: LinearLayout
    private lateinit var tvAnalysisResult: TextView
    private lateinit var tvConfidenceScores: TextView
    private lateinit var tvProcessingTime: TextView
    
    // Service state
    private var deepfakeService: DeepfakeDetectionService? = null
    private var isServiceBound = false
    private var selectedAudioUri: Uri? = null
    
    // Permission handling
    private val permissionLauncher = registerForActivityResult(
        ActivityResultContracts.RequestMultiplePermissions()
    ) { permissions ->
        updatePermissionStatus()
        if (hasAllPermissions()) checkOverlayPermission()
    }
    
    private val overlayPermissionLauncher = registerForActivityResult(
        ActivityResultContracts.StartActivityForResult()
    ) { 
        updatePermissionStatus() 
    }
    
    // File picker for audio analysis
    private val audioFilePickerLauncher = registerForActivityResult(
        ActivityResultContracts.GetContent()
    ) { uri ->
        uri?.let {
            selectedAudioUri = it
            updateSelectedFileUI()
        }
    }
    
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        
        // Setup debug logging
        if (BuildConfig.DEBUG) Timber.plant(Timber.DebugTree())
        
        setContentView(R.layout.activity_main)
        
        initializeViews()
        setupClickListeners()
        updatePermissionStatus()
        
        Timber.i("MainActivity created")
    }
    
    override fun onResume() {
        super.onResume()
        updateServiceStatus()
        updatePermissionStatus()
    }
    
    override fun onDestroy() {
        super.onDestroy()
        if (isServiceBound) {
            unbindService(this)
            isServiceBound = false
        }
    }
    
    private fun initializeViews() {
        tvStatus = findViewById(R.id.tvStatus)
        tvServiceStatus = findViewById(R.id.tvServiceStatus)
        tvModelStatus = findViewById(R.id.tvModelStatus)
        btnToggleService = findViewById(R.id.btnToggleService)
        btnPermissions = findViewById(R.id.btnPermissions)
        btnSettings = findViewById(R.id.btnSettings)
        progressBar = findViewById(R.id.progressBar)
        switchAutoStart = findViewById(R.id.switchAutoStart)
        
        // Audio file analysis UI
        btnSelectAudioFile = findViewById(R.id.btnSelectAudioFile)
        btnAnalyzeFile = findViewById(R.id.btnAnalyzeFile)
        tvSelectedFile = findViewById(R.id.tvSelectedFile)
        layoutAnalysisResults = findViewById(R.id.layoutAnalysisResults)
        tvAnalysisResult = findViewById(R.id.tvAnalysisResult)
        tvConfidenceScores = findViewById(R.id.tvConfidenceScores)
        tvProcessingTime = findViewById(R.id.tvProcessingTime)
        
        // Restore auto-start setting
        val prefs = getSharedPreferences("deepfake_guard", Context.MODE_PRIVATE)
        switchAutoStart.isChecked = prefs.getBoolean("auto_start", false)
    }
    
    private fun setupClickListeners() {
        btnToggleService.setOnClickListener {
            toggleService()
        }
        
        btnPermissions.setOnClickListener {
            requestPermissions()
        }
        
        btnSettings.setOnClickListener {
            showSettingsDialog()
        }
        
        switchAutoStart.setOnCheckedChangeListener { _, isChecked ->
            // Save auto-start preference
            getSharedPreferences("deepfake_guard", Context.MODE_PRIVATE)
                .edit().putBoolean("auto_start", isChecked).apply()
        }
        
        // Audio file analysis click listeners
        btnSelectAudioFile.setOnClickListener {
            selectAudioFile()
        }
        
        btnAnalyzeFile.setOnClickListener {
            analyzeSelectedAudioFile()
        }
    }
    
    private fun hasAllPermissions(): Boolean {
        return REQUIRED_PERMISSIONS.all { permission ->
            ContextCompat.checkSelfPermission(this, permission) == PackageManager.PERMISSION_GRANTED
        }
    }
    
    private fun hasOverlayPermission(): Boolean {
        return if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.M) {
            Settings.canDrawOverlays(this)
        } else {
            true
        }
    }
    
    private fun requestPermissions() {
        val missingPermissions = REQUIRED_PERMISSIONS.filter { permission ->
            ContextCompat.checkSelfPermission(this, permission) != PackageManager.PERMISSION_GRANTED
        }
        
        if (missingPermissions.isNotEmpty()) {
            // Show rationale dialog
            AlertDialog.Builder(this)
                .setTitle("Permissions Required")
                .setMessage("DeepfakeGuard needs these permissions to monitor phone calls and detect deepfakes in real-time:\n\n" +
                        "• Phone permissions: To detect when calls start/end\n" +
                        "• Audio recording: To analyze call audio\n" +
                        "• Foreground service: To run in background")
                .setPositiveButton("Grant") { _, _ ->
                    permissionLauncher.launch(missingPermissions.toTypedArray())
                }
                .setNegativeButton("Cancel", null)
                .show()
        } else {
            checkOverlayPermission()
        }
    }
    
    private fun checkOverlayPermission() {
        if (!hasOverlayPermission()) {
            AlertDialog.Builder(this)
                .setTitle("Overlay Permission Required")
                .setMessage("DeepfakeGuard needs permission to display detection results over other apps during calls.")
                .setPositiveButton("Grant") { _, _ ->
                    val intent = Intent(Settings.ACTION_MANAGE_OVERLAY_PERMISSION).apply {
                        data = Uri.parse("package:$packageName")
                    }
                    overlayPermissionLauncher.launch(intent)
                }
                .setNegativeButton("Skip", null)
                .show()
        }
    }
    
    private fun updatePermissionStatus() {
        val hasPermissions = hasAllPermissions()
        val hasOverlay = hasOverlayPermission()
        
        when {
            hasPermissions && hasOverlay -> {
                tvStatus.text = "✅ All permissions granted"
                tvStatus.setTextColor(ContextCompat.getColor(this, android.R.color.holo_green_dark))
                btnPermissions.text = "Permissions ✓"
                btnPermissions.isEnabled = false
                btnToggleService.isEnabled = true
            }
            hasPermissions -> {
                tvStatus.text = "⚠️ Overlay permission needed"
                tvStatus.setTextColor(ContextCompat.getColor(this, android.R.color.holo_orange_dark))
                btnPermissions.text = "Grant Overlay Permission"
                btnPermissions.isEnabled = true
                btnToggleService.isEnabled = true
            }
            else -> {
                tvStatus.text = "❌ Permissions required"
                tvStatus.setTextColor(ContextCompat.getColor(this, android.R.color.holo_red_dark))
                btnPermissions.text = "Grant Permissions"
                btnPermissions.isEnabled = true
                btnToggleService.isEnabled = false
            }
        }
    }
    
    private fun updateServiceStatus() {
        lifecycleScope.launch {
            val isServiceRunning = isServiceRunning()
            
            if (isServiceRunning) {
                tvServiceStatus.text = "🟢 Service Active - Monitoring calls\n💡 App will automatically optimize audio capture for both callers"
                tvServiceStatus.setTextColor(ContextCompat.getColor(this@MainActivity, android.R.color.holo_green_dark))
                btnToggleService.text = "Stop Monitoring"
                
                // Bind to service to get detailed status
                if (!isServiceBound) {
                    bindToService()
                } else {
                    // Update with audio source info if available
                    deepfakeService?.let { service ->
                        val audioInfo = service.getAudioSourceInfo()
                        val callRecordingInfo = service.getCallRecordingInfo()
                        if (audioInfo != "Not recording") {
                            tvServiceStatus.text = "🟢 Service Active - Monitoring calls\n$audioInfo\n$callRecordingInfo"
                        }
                    }
                }
            } else {
                tvServiceStatus.text = "🔴 Service Stopped"
                tvServiceStatus.setTextColor(ContextCompat.getColor(this@MainActivity, android.R.color.holo_red_dark))
                btnToggleService.text = "Start Monitoring"
                tvModelStatus.text = "Model not loaded"
            }
        }
    }
    
    private fun isServiceRunning(): Boolean {
        val activityManager = getSystemService(Context.ACTIVITY_SERVICE) as android.app.ActivityManager
        @Suppress("DEPRECATION")
        val services = activityManager.getRunningServices(Integer.MAX_VALUE)
        
        return services.any { serviceInfo ->
            serviceInfo.service.className == DeepfakeDetectionService::class.java.name
        }
    }
    
    private fun toggleService() {
        if (isServiceRunning()) {
            stopMonitoringService()
        } else {
            startMonitoringService()
        }
    }
    
    private fun startMonitoringService() {
        if (!hasAllPermissions()) {
            requestPermissions()
            return
        }
        
        val intent = Intent(this, DeepfakeDetectionService::class.java).apply {
            action = DeepfakeDetectionService.ACTION_PREPARE
        }
        
        try {
            startForegroundService(intent)
            Toast.makeText(this, "Starting deepfake monitoring...", Toast.LENGTH_SHORT).show()
            
            // Update UI immediately
            progressBar.visibility = android.view.View.VISIBLE
            btnToggleService.isEnabled = false
            
            // Update status after a delay
            lifecycleScope.launch {
                kotlinx.coroutines.delay(1000)
                updateServiceStatus()
                progressBar.visibility = android.view.View.GONE
                btnToggleService.isEnabled = true
            }
            
        } catch (e: Exception) {
            Timber.e(e, "Failed to start service")
            Toast.makeText(this, "Failed to start service: ${e.message}", Toast.LENGTH_LONG).show()
        }
    }
    
    private fun stopMonitoringService() {
        val intent = Intent(this, DeepfakeDetectionService::class.java)
        stopService(intent)
        
        if (isServiceBound) {
            unbindService(this)
            isServiceBound = false
            deepfakeService = null
        }
        
        Toast.makeText(this, "Stopping deepfake monitoring...", Toast.LENGTH_SHORT).show()
        updateServiceStatus()
    }
    
    private fun bindToService() {
        val intent = Intent(this, DeepfakeDetectionService::class.java)
        bindService(intent, this, Context.BIND_AUTO_CREATE)
    }
    
    override fun onServiceConnected(name: ComponentName?, service: IBinder?) {
        val binder = service as DeepfakeDetectionService.LocalBinder
        deepfakeService = binder.getService()
        isServiceBound = true
        
        // Update model status
        updateModelStatus()
        
        Timber.d("Service connected")
    }
    
    override fun onServiceDisconnected(name: ComponentName?) {
        deepfakeService = null
        isServiceBound = false
        Timber.d("Service disconnected")
    }
    
    /**
     * PUBLIC API: Analyze raw audio data for deepfake detection
     * 
     * This method provides external access to deepfake detection functionality.
     * The service must be running for analysis to work.
     * 
     * @param audioData Raw 16-bit PCM stereo audio data (interleaved L/R samples)
     * @param sampleRate Sample rate of the audio (default: 16000 Hz)
     * @param audioLengthMs Duration of audio in milliseconds (-1 for auto-calculate)
     * @param callback Callback to receive the analysis result
     */
    fun analyzeRawAudio(
        audioData: ShortArray,
        sampleRate: Int = 16000,
        audioLengthMs: Long = -1L,
        callback: (DeepfakeDetectionService.AudioAnalysisResult) -> Unit
    ) {
        lifecycleScope.launch {
            try {
                if (!isServiceBound || deepfakeService == null) {
                    // Try to start and bind to service first
                    if (!isServiceRunning()) {
                        startMonitoringService()
                        // Wait for service to start
                        var attempts = 0
                        while (!isServiceBound && attempts < 50) { // 5 seconds max
                            kotlinx.coroutines.delay(100)
                            attempts++
                        }
                    } else {
                        bindToService()
                        // Wait for binding
                        var attempts = 0
                        while (!isServiceBound && attempts < 30) { // 3 seconds max
                            kotlinx.coroutines.delay(100)
                            attempts++
                        }
                    }
                }
                
                // Wait for model to load
                if (isServiceBound && deepfakeService != null) {
                    var modelAttempts = 0
                    while (!deepfakeService!!.isReadyForAnalysis() && modelAttempts < 100) { // 10s timeout
                        Timber.d("Waiting for model... attempt $modelAttempts")
                        kotlinx.coroutines.delay(100)
                        modelAttempts++
                    }
                    
                    if (!deepfakeService!!.isReadyForAnalysis()) {
                        Timber.w("Model not ready after timeout")
                    }
                }
                
                if (isServiceBound && deepfakeService != null) {
                    val result = deepfakeService!!.analyzeRawAudio(audioData, sampleRate, audioLengthMs)
                    callback(result)
                } else {
                    callback(
                        DeepfakeDetectionService.AudioAnalysisResult(
                            isFake = false,
                            confidence = 0f,
                            fakeConfidence = 0f,
                            realConfidence = 1f,
                            processingTimeMs = 0L,
                            audioLengthMs = audioLengthMs,
                            error = "Service not available. Please check permissions and try again."
                        )
                    )
                }
            } catch (e: Exception) {
                Timber.e(e, "Error in analyzeRawAudio")
                callback(
                    DeepfakeDetectionService.AudioAnalysisResult(
                        isFake = false,
                        confidence = 0f,
                        fakeConfidence = 0f,
                        realConfidence = 1f,
                        processingTimeMs = 0L,
                        audioLengthMs = audioLengthMs,
                        error = "Analysis failed: ${e.message}"
                    )
                )
            }
        }
    }
    
    /**
     * PUBLIC API: Analyze raw audio from ByteArray
     * @param audioBytes 16-bit PCM format
     * @param sampleRate Default 16kHz
     * @param audioLengthMs Duration in ms (-1 = auto)
     * @param callback Result handler
     */
    fun analyzeRawAudio(
        audioBytes: ByteArray,
        sampleRate: Int = 16000,
        audioLengthMs: Long = -1L,
        callback: (DeepfakeDetectionService.AudioAnalysisResult) -> Unit
    ) {
        lifecycleScope.launch {
            try {
                if (!isServiceBound || deepfakeService == null) {
                    // Auto-start and bind to service
                    if (!isServiceRunning()) {
                        startMonitoringService()
                        var attempts = 0
                        while (!isServiceBound && attempts < 50) { // 5s timeout
                            kotlinx.coroutines.delay(100)
                            attempts++
                        }
                    } else {
                        bindToService()
                        var attempts = 0
                        while (!isServiceBound && attempts < 30) { // 3s timeout
                            kotlinx.coroutines.delay(100)
                            attempts++
                        }
                    }
                }
                
                // Wait for model to load
                if (isServiceBound && deepfakeService != null) {
                    var modelAttempts = 0
                    while (!deepfakeService!!.isReadyForAnalysis() && modelAttempts < 100) { // 10s timeout
                        Timber.d("Waiting for model... attempt $modelAttempts")
                        kotlinx.coroutines.delay(100)
                        modelAttempts++
                    }
                    
                    if (!deepfakeService!!.isReadyForAnalysis()) {
                        Timber.w("Model not ready after timeout")
                    }
                }
                
                if (isServiceBound && deepfakeService != null) {
                    val result = deepfakeService!!.analyzeRawAudio(audioBytes, sampleRate, audioLengthMs)
                    callback(result)
                } else {
                    callback(
                        DeepfakeDetectionService.AudioAnalysisResult(
                            isFake = false,
                            confidence = 0f,
                            fakeConfidence = 0f,
                            realConfidence = 1f,
                            processingTimeMs = 0L,
                            audioLengthMs = audioLengthMs,
                            error = "Service not available. Please check permissions and try again."
                        )
                    )
                }
            } catch (e: Exception) {
                Timber.e(e, "Error in analyzeRawAudio (ByteArray)")
                callback(
                    DeepfakeDetectionService.AudioAnalysisResult(
                        isFake = false,
                        confidence = 0f,
                        fakeConfidence = 0f,
                        realConfidence = 1f,
                        processingTimeMs = 0L,
                        audioLengthMs = audioLengthMs,
                        error = "Analysis failed: ${e.message}"
                    )
                )
            }
        }
    }
    
    /** Check if service is ready for analysis */
    fun isReadyForAnalysis(): Boolean = isServiceBound && deepfakeService?.isReadyForAnalysis() == true

    
    // === Audio File Analysis ===
    
    private fun selectAudioFile() {
        try {
            audioFilePickerLauncher.launch("audio/*")
        } catch (e: Exception) {
            Timber.e(e, "Error launching file picker")
            Toast.makeText(this, "Error opening file picker: ${e.message}", Toast.LENGTH_SHORT).show()
        }
    }
    
    private fun updateSelectedFileUI() {
        selectedAudioUri?.let { uri ->
            try {
                val fileName = getFileName(uri)
                tvSelectedFile.text = fileName
                btnAnalyzeFile.isEnabled = true
                
                // Hide previous results
                layoutAnalysisResults.visibility = View.GONE
                
                Timber.d("Selected audio file: $fileName")
            } catch (e: Exception) {
                Timber.e(e, "Error updating selected file UI")
                tvSelectedFile.text = "Error reading file"
                btnAnalyzeFile.isEnabled = false
            }
        }
    }
    
    private fun getFileName(uri: Uri): String {
        return try {
            contentResolver.query(uri, null, null, null, null)?.use { cursor ->
                val nameIndex = cursor.getColumnIndex(android.provider.OpenableColumns.DISPLAY_NAME)
                cursor.moveToFirst()
                cursor.getString(nameIndex)
            } ?: "Unknown file"
        } catch (e: Exception) {
            "Audio file"
        }
    }
    
    private fun analyzeSelectedAudioFile() {
        selectedAudioUri?.let { uri ->
            lifecycleScope.launch {
                try {
                    btnAnalyzeFile.isEnabled = false
                    progressBar.visibility = View.VISIBLE
                    
                    // Check if service is running, if not start it
                    if (!isServiceRunning()) {
                        btnAnalyzeFile.text = "🔄 Starting Service..."
                        Toast.makeText(this@MainActivity, "Starting deepfake detection service...", Toast.LENGTH_SHORT).show()
                    } else {
                        btnAnalyzeFile.text = "🔄 Loading Model..."
                    }
                    
                    // Read audio file and convert to audio data
                    btnAnalyzeFile.text = "🔄 Reading Audio..."
                    val audioData = readAudioFile(uri)
                    
                    if (audioData != null) {
                        btnAnalyzeFile.text = "🔄 Analyzing..."
                        
                        // Analyze the audio
                        analyzeRawAudio(audioData) { result ->
                            lifecycleScope.launch {
                                displayAnalysisResult(result)
                                btnAnalyzeFile.isEnabled = true
                                btnAnalyzeFile.text = "🔍 Analyze Audio"
                                progressBar.visibility = View.GONE
                            }
                        }
                    } else {
                        Toast.makeText(this@MainActivity, "Failed to read audio file", Toast.LENGTH_SHORT).show()
                        btnAnalyzeFile.isEnabled = true
                        btnAnalyzeFile.text = "🔍 Analyze Audio"
                        progressBar.visibility = View.GONE
                    }
                    
                } catch (e: Exception) {
                    Timber.e(e, "Error analyzing audio file")
                    Toast.makeText(this@MainActivity, "Error analyzing file: ${e.message}", Toast.LENGTH_LONG).show()
                    btnAnalyzeFile.isEnabled = true
                    btnAnalyzeFile.text = "🔍 Analyze Audio"
                    progressBar.visibility = View.GONE
                }
            }
        }
    }
    
    private suspend fun readAudioFile(uri: Uri): ShortArray? {
        return try {
            // Basic WAV/PCM file reader
            contentResolver.openInputStream(uri)?.use { inputStream ->
                val bytes = inputStream.readBytes()
                
                // Skip WAV header if present (44 bytes)
                val dataStartIndex = if (bytes.size > 44 && 
                    bytes.sliceArray(0..3).contentEquals("RIFF".toByteArray())) {
                    44 // Skip WAV header
                } else {
                    0 // Raw PCM
                }
                
                // Convert to 16-bit PCM samples
                val audioBytes = bytes.sliceArray(dataStartIndex until bytes.size)
                val samples = ShortArray(audioBytes.size / 2)
                
                for (i in samples.indices) {
                    val low = audioBytes[i * 2].toInt() and 0xFF
                    val high = audioBytes[i * 2 + 1].toInt() and 0xFF
                    samples[i] = ((high shl 8) or low).toShort()
                }
                
                Timber.d("Read audio file: ${samples.size} samples")
                samples
            }
        } catch (e: Exception) {
            Timber.e(e, "Error reading audio file")
            null
        }
    }
    
    private fun displayAnalysisResult(result: DeepfakeDetectionService.AudioAnalysisResult) {
        try {
            layoutAnalysisResults.visibility = View.VISIBLE
            
            if (result.error != null) {
                tvAnalysisResult.text = "❌ Analysis Error"
                tvAnalysisResult.setTextColor(ContextCompat.getColor(this, R.color.error_color))
                tvConfidenceScores.text = "Error: ${result.error}"
                tvProcessingTime.text = ""
            } else {
                // Display main result
                if (result.isFake) {
                    tvAnalysisResult.text = "🚨 DEEPFAKE DETECTED"
                    tvAnalysisResult.setTextColor(ContextCompat.getColor(this, R.color.error_color))
                } else {
                    tvAnalysisResult.text = "✅ Real Audio"
                    tvAnalysisResult.setTextColor(ContextCompat.getColor(this, R.color.success_color))
                }
                
                // Display confidence scores
                val fakePercent = (result.fakeConfidence * 100).toInt()
                val realPercent = (result.realConfidence * 100).toInt()
                tvConfidenceScores.text = "Real: $realPercent% | Fake: $fakePercent%"
                
                // Display processing time and audio length
                val audioLengthText = if (result.audioLengthMs > 0) {
                    " | Audio: ${result.audioLengthMs}ms"
                } else ""
                tvProcessingTime.text = "Processing: ${result.processingTimeMs}ms$audioLengthText"
                
                Timber.i("Analysis complete: ${if (result.isFake) "FAKE" else "REAL"} (${result.confidence * 100}% confidence)")
            }
        } catch (e: Exception) {
            Timber.e(e, "Error displaying analysis result")
        }
    }
    
    private fun updateModelStatus() {
        // Update model status indicator
        tvModelStatus.text = "Model loaded ✓"
        tvModelStatus.setTextColor(ContextCompat.getColor(this, android.R.color.holo_green_dark))
    }
    
    private fun showSettingsDialog() {
        val options = arrayOf(
            "Notification Settings",
            "About"
        )
        
        AlertDialog.Builder(this)
            .setTitle("Settings")
            .setItems(options) { _, which ->
                when (which) {
                    0 -> showNotificationSettings()
                    1 -> showAbout()
                }
            }
            .show()
    }
    
    private fun showNotificationSettings() {
        val intent = Intent(Settings.ACTION_APP_NOTIFICATION_SETTINGS).apply {
            putExtra(Settings.EXTRA_APP_PACKAGE, packageName)
        }
        startActivity(intent)
    }
    
    private fun showAbout() {
        AlertDialog.Builder(this)
            .setTitle("About DeepfakeGuard")
            .setMessage("""
DeepfakeGuard v1.0
                
                Real-time deepfake audio detection for calls.
                Uses on-device AI to protect against audio scams.
                
                Features:
                • Real-time call monitoring
                • Offline processing
                • Visual detection alerts
                • File analysis
                
                Built with PyTorch Mobile
            """.trimIndent())
            .setPositiveButton("OK", null)
            .show()
    }
}
