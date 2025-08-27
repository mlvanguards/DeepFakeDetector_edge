/**
 * Foreground service for real-time deepfake detection during calls
 * Captures audio, runs PyTorch inference, displays overlay results
 */
package com.example.deepfakeguard

import android.app.*
import android.content.Context
import android.content.Intent
import android.media.*
import android.os.Binder
import android.os.Build
import android.os.IBinder
import android.view.WindowManager
import androidx.core.app.NotificationCompat
import kotlinx.coroutines.*
import org.pytorch.IValue
import org.pytorch.Module
import org.pytorch.Tensor
import timber.log.Timber
import java.io.File
import java.io.FileOutputStream
import java.util.concurrent.atomic.AtomicBoolean
import kotlin.math.*

class DeepfakeDetectionService : Service() {
    
    companion object {
        const val ACTION_PREPARE = "ACTION_PREPARE"
        const val ACTION_START_DETECTION = "ACTION_START_DETECTION"
        const val ACTION_STOP_DETECTION = "ACTION_STOP_DETECTION"
        const val EXTRA_PHONE_NUMBER = "EXTRA_PHONE_NUMBER"
        const val EXTRA_IS_INCOMING = "EXTRA_IS_INCOMING"
        
        private const val NOTIFICATION_ID = 1001
        private const val CHANNEL_ID = "deepfake_detection_channel"
        
        // Audio processing setup
        private const val SAMPLE_RATE = 16000
        private const val CHANNEL_CONFIG = AudioFormat.CHANNEL_IN_STEREO  // Stereo for CRNN
        private const val AUDIO_FORMAT = AudioFormat.ENCODING_PCM_16BIT
        private const val BUFFER_SIZE_FACTOR = 4
        
        // Model input specs - CRNN mel spectrogram [2, 64, 300]
        private const val MODEL_INPUT_SIZE = 2 * 64 * 300  // 38400 elements
        private const val AUDIO_CHUNK_DURATION_MS = 4000   // 4s chunks (training match)
        private const val OVERLAP_DURATION_MS = 500        // 0.5s overlap
    }
    
    private val binder = LocalBinder()
    private val serviceScope = CoroutineScope(Dispatchers.Main + SupervisorJob())
    
    // Audio processing
    private var audioRecord: AudioRecord? = null
    private val isRecording = AtomicBoolean(false)
    private var audioProcessingJob: Job? = null
    
    // Audio state backup for restoration
    private var originalAudioMode: Int = AudioManager.MODE_NORMAL
    private var originalSpeakerState: Boolean = false
    private var currentAudioSource: Int = MediaRecorder.AudioSource.DEFAULT
    private var originalCommunicationDevice: AudioDeviceInfo? = null
    
    // ML inference
    private var deepfakeModel: Module? = null
    private var isModelLoaded = AtomicBoolean(false)
    private val audioProcessor = AudioProcessor()
    
    // Call state
    private var currentPhoneNumber: String? = null
    private var isIncomingCall = false
    
    // Results tracking
    private var detectionResults = mutableListOf<DetectionResult>()
    private var lastDetectionTime = 0L
    
    // UI overlay
    private var overlayView: OverlayView? = null
    private var windowManager: WindowManager? = null
    
    inner class LocalBinder : Binder() {
        fun getService(): DeepfakeDetectionService = this@DeepfakeDetectionService
    }
    
    data class DetectionResult(
        val timestamp: Long,
        val isFake: Boolean,
        val confidence: Float,
        val audioChunkId: Int = -1,  // -1 for external audio analysis
        val processingTimeMs: Long = 0L
    )
    
    data class AudioAnalysisResult(
        val isFake: Boolean,
        val confidence: Float,
        val fakeConfidence: Float,
        val realConfidence: Float,
        val processingTimeMs: Long,
        val audioLengthMs: Long,
        val error: String? = null
    )
    
    override fun onCreate() {
        super.onCreate()
        createNotificationChannel()
        initializeWindowManager()
        Timber.d("DeepfakeDetectionService created")
    }
    
    override fun onBind(intent: Intent?): IBinder = binder
    
    override fun onStartCommand(intent: Intent?, flags: Int, startId: Int): Int {
        val action = intent?.action ?: return START_NOT_STICKY
        
        when (action) {
            ACTION_PREPARE -> {
                handlePrepare(intent)
            }
            ACTION_START_DETECTION -> {
                handleStartDetection(intent)
            }
            ACTION_STOP_DETECTION -> {
                handleStopDetection()
            }
        }
        
        return START_STICKY
    }
    
    private fun handlePrepare(intent: Intent) {
        currentPhoneNumber = intent.getStringExtra(EXTRA_PHONE_NUMBER)
        isIncomingCall = intent.getBooleanExtra(EXTRA_IS_INCOMING, false)
        
        startForeground(NOTIFICATION_ID, createNotification("Preparing for call monitoring..."))
        
        // Load model asynchronously
        serviceScope.launch(Dispatchers.IO) { loadDeepfakeModel() }
    }
    
    private fun handleStartDetection(intent: Intent) {
        if (!isModelLoaded.get()) {
            Timber.w("Model not loaded yet, preparing...")
            handlePrepare(intent)
            return
        }
        
        currentPhoneNumber = intent.getStringExtra(EXTRA_PHONE_NUMBER)
        isIncomingCall = intent.getBooleanExtra(EXTRA_IS_INCOMING, false)
        
        Timber.i("Starting detection for: $currentPhoneNumber (incoming: $isIncomingCall)")
        
        startForeground(NOTIFICATION_ID, createNotification("Monitoring call for deepfakes..."))
        showOverlay()
        startAudioMonitoring()
    }
    
    private fun handleStopDetection() {
        Timber.i("Stopping detection")
        
        stopAudioMonitoring()
        hideOverlay()
        
        generateCallSummary() // Log call stats
        
        stopForeground(STOP_FOREGROUND_REMOVE)
        stopSelf()
    }
    
    private fun loadDeepfakeModel() {
        try {
            Timber.i("Loading ML model...")
            
            val modelFile = getModelFile()
            Timber.d("Model file path: ${modelFile.absolutePath}")
            Timber.d("Model file exists: ${modelFile.exists()}")
            Timber.d("Model file size: ${modelFile.length()} bytes")
            
            if (!modelFile.exists()) {
                Timber.e("Model file not found: ${modelFile.absolutePath}")
                isModelLoaded.set(false)
                return
            }
            
            if (modelFile.length() == 0L) {
                Timber.e("Model file is empty: ${modelFile.absolutePath}")
                isModelLoaded.set(false)
                return
            }
            
            Timber.d("Loading PyTorch model from file...")
            deepfakeModel = Module.load(modelFile.absolutePath)
            isModelLoaded.set(true)
            
            Timber.i("✅ ML model loaded successfully!")
            Timber.d("Model ready: ${isReadyForAnalysis()}")
            
            // Update notification
            serviceScope.launch(Dispatchers.Main) {
                val notification = createNotification("Model loaded - Ready to analyze audio")
                val notificationManager = getSystemService(Context.NOTIFICATION_SERVICE) as NotificationManager
                notificationManager.notify(NOTIFICATION_ID, notification)
            }
            
        } catch (e: Exception) {
            Timber.e(e, "❌ Failed to load ML model")
            isModelLoaded.set(false)
            
            // Update notification with error
            serviceScope.launch(Dispatchers.Main) {
                val notification = createNotification("Model loading failed - Check logs")
                val notificationManager = getSystemService(Context.NOTIFICATION_SERVICE) as NotificationManager
                notificationManager.notify(NOTIFICATION_ID, notification)
            }
        }
    }
    
    private fun getModelFile(): File {
        // Check for model in internal storage
        val assetsModelFile = File(filesDir, "deepfake_detector.pt")
        
        Timber.d("Model path: ${assetsModelFile.absolutePath}")
        Timber.d("Exists: ${assetsModelFile.exists()}")
        
        if (!assetsModelFile.exists()) {
            // Copy from assets to internal storage
            try {
                Timber.d("Copying from assets/models/deepfake_detector.pt...")
                
                assets.open("models/deepfake_detector.pt").use { input ->
                    FileOutputStream(assetsModelFile).use { output ->
                        val bytesCopied = input.copyTo(output)
                        Timber.d("Copied $bytesCopied bytes from assets")
                    }
                }
                
                Timber.i("✅ Model copied from assets to: ${assetsModelFile.absolutePath}")
                Timber.d("Copied file size: ${assetsModelFile.length()} bytes")
                
            } catch (e: Exception) {
                Timber.e(e, "❌ Failed to copy model from assets")
            }
        } else {
            Timber.d("Model file already exists, size: ${assetsModelFile.length()} bytes")
        }
        
        return assetsModelFile
    }
    
    private fun startAudioMonitoring() {
        if (isRecording.get()) {
            Timber.w("Audio monitoring already active")
            return
        }
        
        try {
            val bufferSize = AudioRecord.getMinBufferSize(SAMPLE_RATE, CHANNEL_CONFIG, AUDIO_FORMAT) * BUFFER_SIZE_FACTOR
            
            // Audio sources in priority order
            val audioSources = arrayOf(
                MediaRecorder.AudioSource.VOICE_DOWNLINK,       // 🎯 Other caller only
                MediaRecorder.AudioSource.VOICE_CALL,           // Full call audio
                MediaRecorder.AudioSource.VOICE_UPLINK,         // Your voice only  
                MediaRecorder.AudioSource.VOICE_COMMUNICATION,  // Call-optimized mic
                MediaRecorder.AudioSource.UNPROCESSED,          // Raw microphone
                MediaRecorder.AudioSource.MIC,                  // Basic microphone
                MediaRecorder.AudioSource.DEFAULT               // Fallback
            )
            
            var tempAudioRecord: AudioRecord? = null
            var lastError: String? = null
            
            for (audioSource in audioSources) {
                try {
                    val sourceName = when(audioSource) {
                        MediaRecorder.AudioSource.VOICE_CALL -> "VOICE_CALL (both)"
                        MediaRecorder.AudioSource.VOICE_DOWNLINK -> "VOICE_DOWNLINK 🎯"
                        MediaRecorder.AudioSource.VOICE_UPLINK -> "VOICE_UPLINK (you)"
                        MediaRecorder.AudioSource.VOICE_COMMUNICATION -> "VOICE_COMM"
                        MediaRecorder.AudioSource.UNPROCESSED -> "UNPROCESSED"
                        MediaRecorder.AudioSource.MIC -> "MIC"
                        MediaRecorder.AudioSource.DEFAULT -> "DEFAULT"
                        else -> "UNKNOWN ($audioSource)"
                    }
                    Timber.d("🎙️ Trying: $sourceName")
                    
                    // For call-specific sources, try different configurations
                    if (audioSource == MediaRecorder.AudioSource.VOICE_DOWNLINK || 
                        audioSource == MediaRecorder.AudioSource.VOICE_CALL ||
                        audioSource == MediaRecorder.AudioSource.VOICE_UPLINK) {
                        
                        // Try different channel configs
                        val channelConfigs = arrayOf(
                            AudioFormat.CHANNEL_IN_STEREO,  // Preferred
                            AudioFormat.CHANNEL_IN_MONO,   // Fallback
                            CHANNEL_CONFIG                  // Original
                        )
                        
                        for (channelConfig in channelConfigs) {
                            try {
                                val adjustedBufferSize = AudioRecord.getMinBufferSize(SAMPLE_RATE, channelConfig, AUDIO_FORMAT) * BUFFER_SIZE_FACTOR
                                if (adjustedBufferSize > 0) {
                                    Timber.d("🔧 $sourceName config: $channelConfig")
                                    
                                    tempAudioRecord = AudioRecord(
                                        audioSource,
                                        SAMPLE_RATE,
                                        channelConfig,
                                        AUDIO_FORMAT,
                                        adjustedBufferSize
                                    )
                                    
                                    if (tempAudioRecord.state == AudioRecord.STATE_INITIALIZED) {
                                        Timber.i("✅ SUCCESS: $sourceName")
                                        break  // Success! Use this configuration
                                    } else {
                                        tempAudioRecord.release()
                                        tempAudioRecord = null
                                    }
                                }
                            } catch (e: Exception) {
                                Timber.w("Channel config $channelConfig failed for $sourceName: ${e.message}")
                                tempAudioRecord?.release()
                                tempAudioRecord = null
                            }
                        }
                    } else {
                        // Use standard configuration for non-call sources
                        tempAudioRecord = AudioRecord(
                            audioSource,
                            SAMPLE_RATE,
                            CHANNEL_CONFIG,
                            AUDIO_FORMAT,
                            bufferSize
                        )
                    }
                    
                    if (tempAudioRecord != null && tempAudioRecord.state == AudioRecord.STATE_INITIALIZED) {
                        Timber.i("✅ AudioRecord initialized successfully with source: $sourceName")
                        this.audioRecord = tempAudioRecord
                        
                        // Store which source we're using for optimization
                        currentAudioSource = audioSource
                        break
                    } else {
                        Timber.w("AudioRecord state not initialized for source: $audioSource")
                        tempAudioRecord?.release()
                        tempAudioRecord = null
                    }
                } catch (e: Exception) {
                    Timber.w(e, "Failed to initialize AudioRecord with source: $audioSource")
                    lastError = e.message
                    tempAudioRecord?.release()
                    tempAudioRecord = null
                }
            }
            
            if (this.audioRecord == null) {
                Timber.e("Failed to initialize AudioRecord with any audio source. Last error: $lastError")
                
                // Show recording unavailable
                serviceScope.launch(Dispatchers.Main) {
                    val notification = createNotification("Call recording not available")
                    val notificationManager = getSystemService(Context.NOTIFICATION_SERVICE) as NotificationManager
                    notificationManager.notify(NOTIFICATION_ID, notification)
                }
                return
            }
            
            this.audioRecord?.startRecording()
            isRecording.set(true)
            
            // Configure audio settings for better call recording
            val audioManager = getSystemService(Context.AUDIO_SERVICE) as AudioManager
            
            // Save original settings for restoration later
            originalAudioMode = audioManager.mode
            originalSpeakerState = isSpeakerphoneActive()
            
            // Optimize for call recording based on audio source
            try {
                audioManager.mode = AudioManager.MODE_IN_COMMUNICATION
                
                // If we can't get direct call audio, enable speakerphone to capture both sides
                val needsSpeakerphone = when(currentAudioSource) {
                    MediaRecorder.AudioSource.VOICE_CALL,      // Direct call audio - no need
                    MediaRecorder.AudioSource.VOICE_DOWNLINK,  // Other caller only - no need  
                    MediaRecorder.AudioSource.VOICE_UPLINK -> false // Your voice only - no need
                    
                    MediaRecorder.AudioSource.VOICE_COMMUNICATION, // Mic for calls - help with speaker
                    MediaRecorder.AudioSource.UNPROCESSED,         // Raw mic - help with speaker
                    MediaRecorder.AudioSource.MIC,                 // Basic mic - help with speaker
                    MediaRecorder.AudioSource.DEFAULT -> true      // Fallback - help with speaker
                    
                    else -> true
                }
                
                if (needsSpeakerphone && !originalSpeakerState) {
                                                setSpeakerphoneOn(true)
                    Timber.i("🔊 Enabled speaker for better capture (source: $currentAudioSource)")
                    
                                            // Optimize audio settings
                        try {
                            val maxVolume = audioManager.getStreamMaxVolume(AudioManager.STREAM_VOICE_CALL)
                            audioManager.setStreamVolume(AudioManager.STREAM_VOICE_CALL, maxVolume, 0)
                            
                            audioManager.requestAudioFocus(
                                null,
                                AudioManager.STREAM_VOICE_CALL,
                                AudioManager.AUDIOFOCUS_GAIN
                            )
                            
                            Timber.d("📢 Audio optimized")
                        } catch (e: Exception) {
                            Timber.w(e, "Audio optimization failed")
                        }
                }
                
                Timber.d("Audio: mode=${audioManager.mode}, speaker=${isSpeakerphoneActive()}, source=$currentAudioSource")
                Timber.d("Saved: mode=$originalAudioMode, speaker=$originalSpeakerState")
            } catch (e: Exception) {
                Timber.w(e, "Failed to optimize audio settings")
            }
            
            // Start audio processing in background
            audioProcessingJob = serviceScope.launch(Dispatchers.IO) {
                processAudioStream()
            }
            
            Timber.i("✅ Audio monitoring started successfully")
            
            // Update notification with status
            serviceScope.launch(Dispatchers.Main) {
                val sourceDesc = when(currentAudioSource) {
                    MediaRecorder.AudioSource.VOICE_CALL -> "🔴 Full call"
                    MediaRecorder.AudioSource.VOICE_DOWNLINK -> "🔴 Other caller"
                    MediaRecorder.AudioSource.VOICE_UPLINK -> "🔴 Your voice"
                    MediaRecorder.AudioSource.VOICE_COMMUNICATION -> "🔴 Microphone"
                    else -> "🔴 Call audio"
                }
                
                val speakerInfo = if (isSpeakerphoneActive()) " (speaker)" else ""
                val notification = createNotification("$sourceDesc$speakerInfo - Detection active")
                val notificationManager = getSystemService(Context.NOTIFICATION_SERVICE) as NotificationManager
                notificationManager.notify(NOTIFICATION_ID, notification)
            }
            
        } catch (e: Exception) {
            Timber.e(e, "Failed to start audio monitoring")
        }
    }
    
    private fun stopAudioMonitoring() {
        isRecording.set(false)
        
        audioProcessingJob?.cancel()
        audioProcessingJob = null
        
        audioRecord?.apply {
            try {
                stop()
                release()
            } catch (e: Exception) {
                Timber.e(e, "Error stopping AudioRecord")
            }
        }
        audioRecord = null
        
        // Restore original settings
        try {
            val audioManager = getSystemService(Context.AUDIO_SERVICE) as AudioManager
            audioManager.mode = originalAudioMode
            setSpeakerphoneOn(originalSpeakerState)
            Timber.d("Audio restored: mode=$originalAudioMode, speaker=$originalSpeakerState")
        } catch (e: Exception) {
            Timber.w(e, "Failed to restore audio")
        }
        
        Timber.i("Audio monitoring stopped")
    }
    
    private suspend fun processAudioStream() {
        val chunkSamples = (SAMPLE_RATE * AUDIO_CHUNK_DURATION_MS) / 1000
        val overlapSamples = (SAMPLE_RATE * OVERLAP_DURATION_MS) / 1000
        val bufferSize = AudioRecord.getMinBufferSize(SAMPLE_RATE, CHANNEL_CONFIG, AUDIO_FORMAT)
        
        val audioBuffer = ShortArray(bufferSize)
        val audioChunk = mutableListOf<Short>()
        var chunkId = 0
        
        Timber.d("Audio loop: $chunkSamples samples/chunk")
        
        while (isRecording.get() && currentCoroutineContext().isActive) {
            try {
                val bytesRead = audioRecord?.read(audioBuffer, 0, bufferSize) ?: 0
                
                if (bytesRead > 0) {
                    // Add new audio data to chunk
                    for (i in 0 until bytesRead) {
                        audioChunk.add(audioBuffer[i])
                    }
                    
                    // Process chunk when we have enough samples
                    if (audioChunk.size >= chunkSamples) {
                        val chunkToProcess = audioChunk.take(chunkSamples).toShortArray()
                        
                        // Process this chunk for deepfake detection
                        processAudioChunk(chunkToProcess, chunkId++)
                        
                        // Remove processed samples (keep overlap)
                        val samplesToRemove = chunkSamples - overlapSamples
                        repeat(samplesToRemove) {
                            if (audioChunk.isNotEmpty()) {
                                audioChunk.removeAt(0)
                            }
                        }
                    }
                }
                
                // Small delay to prevent excessive CPU usage
                delay(10)
                
            } catch (e: Exception) {
                Timber.e(e, "Error processing audio stream")
                break
            }
        }
        
        Timber.d("Audio processing loop ended")
    }
    
    private suspend fun processAudioChunk(audioData: ShortArray, chunkId: Int) {
        try {
            if (!isModelLoaded.get() || deepfakeModel == null) {
                return
            }
            
            // Monitor audio levels (every 10 chunks)
            if (chunkId % 10 == 0) {
                val max = audioData.maxOrNull()?.toFloat() ?: 0f
                val rms = kotlin.math.sqrt(audioData.map { (it * it).toDouble() }.average()).toFloat()
                val nonZero = audioData.count { it != 0.toShort() }
                Timber.d("📊 Chunk $chunkId: max=$max, rms=$rms, nonZero=$nonZero")
            }
            
            // Generate mel spectrogram from stereo audio chunk
            val melSpectrogram = audioProcessor.generateMelSpectrogram(audioData, SAMPLE_RATE)
            
            if (melSpectrogram.size != MODEL_INPUT_SIZE) {
                Timber.w("Mel spectrogram size mismatch: expected $MODEL_INPUT_SIZE, got ${melSpectrogram.size}")
                return
            }
            
            // Convert mel spectrogram to PyTorch tensor [1, 2, 64, 300]
            val inputTensor = Tensor.fromBlob(
                melSpectrogram, 
                longArrayOf(1, 2, 64, 300)  // [batch, channels, mel_bins, time_steps]
            )
            
            // Run inference
            val outputTensor = deepfakeModel!!.forward(IValue.from(inputTensor)).toTensor()
            val output = outputTensor.dataAsFloatArray
            
            // Apply sigmoid to convert logits to probabilities (from training code)
            val sigmoidOutput = output.map { 1.0f / (1.0f + kotlin.math.exp(-it)) }
            
            // Interpret results - model outputs single value, > 0.5 means fake
            val fakeProb = if (sigmoidOutput.isNotEmpty()) sigmoidOutput[0] else 0f
            val realProb = 1f - fakeProb
            val isFake = fakeProb > 0.5f
            
            val result = DetectionResult(
                timestamp = System.currentTimeMillis(),
                isFake = isFake,
                confidence = fakeProb,
                audioChunkId = chunkId
            )
            
            detectionResults.add(result)
            
            // Update UI overlay
            withContext(Dispatchers.Main) {
                updateOverlay(result)
            }
            
            // Log significant detections
            if (isFake && fakeProb > 0.7f) {
                Timber.w("HIGH CONFIDENCE DEEPFAKE DETECTED: ${fakeProb * 100}% (Real: ${realProb * 100}%)")
            }
            
            lastDetectionTime = System.currentTimeMillis()
            
        } catch (e: Exception) {
            Timber.e(e, "Error processing audio chunk $chunkId")
        }
    }
    
    /**
     * Analyze raw audio data for deepfake detection
     * 
     * @param audioData Raw 16-bit PCM audio data (stereo interleaved)
     * @param sampleRate Sample rate of the audio (default: 16000 Hz)
     * @param audioLengthMs Duration of the audio in milliseconds (for metadata)
     * @return AudioAnalysisResult with detection results and metadata
     */
    suspend fun analyzeRawAudio(
        audioData: ShortArray, 
        sampleRate: Int = SAMPLE_RATE,
        audioLengthMs: Long = -1L
    ): AudioAnalysisResult {
        val startTime = System.currentTimeMillis()
        
        return withContext(Dispatchers.IO) {
            try {
                // Check if model is loaded
                if (!isModelLoaded.get() || deepfakeModel == null) {
                    return@withContext AudioAnalysisResult(
                        isFake = false,
                        confidence = 0f,
                        fakeConfidence = 0f,
                        realConfidence = 1f,
                        processingTimeMs = System.currentTimeMillis() - startTime,
                        audioLengthMs = audioLengthMs,
                        error = "Model not loaded. Please start the service first."
                    )
                }
            
                // Validate input
                if (audioData.isEmpty()) {
                    return@withContext AudioAnalysisResult(
                        isFake = false,
                        confidence = 0f,
                        fakeConfidence = 0f,
                        realConfidence = 1f,
                        processingTimeMs = System.currentTimeMillis() - startTime,
                        audioLengthMs = audioLengthMs,
                        error = "Empty audio data provided"
                    )
                }
            
                // Calculate actual audio length if not provided
                val actualAudioLengthMs = if (audioLengthMs > 0) {
                    audioLengthMs
                } else {
                    ((audioData.size / 2) * 1000L) / sampleRate  // Stereo audio
                }
                
                // Resample to 16kHz if needed (simplified - assumes input is already 16kHz for now)
                val processedAudio = if (sampleRate == SAMPLE_RATE) {
                    audioData
                } else {
                    // TODO: Implement resampling for different sample rates
                    Timber.w("Sample rate $sampleRate != $SAMPLE_RATE. Resampling not implemented yet.")
                    audioData
                }
            
            // Match training preprocessing: ensure stereo and exactly 4 seconds
            val maxLen = SAMPLE_RATE * (AUDIO_CHUNK_DURATION_MS / 1000)  // 4 seconds in samples
            val expectedSamples = maxLen * 2  // Stereo interleaved
            
            // If input is mono (half expected size), convert to stereo by repeating
            val processedStereo = if (processedAudio.size < expectedSamples / 2) {
                // Likely mono, convert to stereo by repeating each sample
                val stereoArray = ShortArray(processedAudio.size * 2)
                for (i in processedAudio.indices) {
                    stereoArray[i * 2] = processedAudio[i]       // Left channel
                    stereoArray[i * 2 + 1] = processedAudio[i]   // Right channel (same)
                }
                Timber.d("Converted mono to stereo: ${processedAudio.size} -> ${stereoArray.size} samples")
                stereoArray
            } else {
                processedAudio
            }
            
            // Pad or trim to exactly 4 seconds stereo
            val processedAudioPadded = when {
                processedStereo.size > expectedSamples -> {
                    // Take first 4 seconds
                    processedStereo.sliceArray(0 until expectedSamples)
                }
                processedStereo.size < expectedSamples -> {
                    // Pad with zeros
                    val padded = ShortArray(expectedSamples)
                    processedStereo.copyInto(padded)
                    padded
                }
                else -> processedStereo
            }
            
                Timber.d("Final audio samples: ${processedAudioPadded.size} (expected: $expectedSamples)")
                
                // Generate mel spectrogram from stereo audio with timeout
                Timber.d("🔄 Starting mel spectrogram generation...")
                val melStartTime = System.currentTimeMillis()
                
                val melSpectrogram = try {
                    withTimeout(30000) { // 30 second timeout
                        audioProcessor.generateMelSpectrogram(processedAudioPadded, SAMPLE_RATE)
                    }
                } catch (e: kotlinx.coroutines.TimeoutCancellationException) {
                    Timber.e("❌ Mel spectrogram generation timed out after 30 seconds")
                    return@withContext AudioAnalysisResult(
                        isFake = false,
                        confidence = 0f,
                        fakeConfidence = 0f,
                        realConfidence = 1f,
                        processingTimeMs = System.currentTimeMillis() - startTime,
                        audioLengthMs = actualAudioLengthMs,
                        error = "Mel spectrogram generation timed out - audio processing too slow"
                    )
                }
                
                val melEndTime = System.currentTimeMillis()
                Timber.d("✅ Mel spectrogram generated in ${melEndTime - melStartTime}ms")
                
                if (melSpectrogram.size != MODEL_INPUT_SIZE) {
                    return@withContext AudioAnalysisResult(
                        isFake = false,
                        confidence = 0f,
                        fakeConfidence = 0f,
                        realConfidence = 1f,
                        processingTimeMs = System.currentTimeMillis() - startTime,
                        audioLengthMs = actualAudioLengthMs,
                        error = "Mel spectrogram size mismatch: expected $MODEL_INPUT_SIZE, got ${melSpectrogram.size}"
                    )
                }
            
                // Convert mel spectrogram to PyTorch tensor [1, 2, 64, 300]
                val inputTensor = Tensor.fromBlob(
                    melSpectrogram, 
                    longArrayOf(1, 2, 64, 300)  // [batch, channels, mel_bins, time_steps]
                )
                
                // Validate input tensor
                Timber.d("Input tensor shape: [${inputTensor.shape().contentToString()}]")
                val inputStats = melSpectrogram.let {
                    val min = it.minOrNull() ?: 0f
                    val max = it.maxOrNull() ?: 0f
                    val mean = it.average().toFloat()
                    "min=$min, max=$max, mean=$mean"
                }
                Timber.d("Input tensor stats: $inputStats")
                
                // Run inference
                Timber.d("🔄 Running model inference...")
                val inferenceStartTime = System.currentTimeMillis()
                
                val outputTensor = deepfakeModel!!.forward(IValue.from(inputTensor)).toTensor()
                val output = outputTensor.dataAsFloatArray
                
                val inferenceEndTime = System.currentTimeMillis()
                Timber.d("✅ Model inference completed in ${inferenceEndTime - inferenceStartTime}ms")
                
                Timber.d("Model output (logits): ${output.joinToString()}")
                
                // Apply sigmoid to convert logits to probabilities (from training code)
                val sigmoidOutput = output.map { 1.0f / (1.0f + kotlin.math.exp(-it)) }
                
                // Interpret results - model outputs single value, > 0.5 means fake
                val fakeProb = if (sigmoidOutput.isNotEmpty()) sigmoidOutput[0] else 0f
                val realProb = 1f - fakeProb
                val isFake = fakeProb > 0.5f
                
                Timber.d("Sigmoid output: ${sigmoidOutput.joinToString()}")
                Timber.d("Probabilities - Real: $realProb, Fake: $fakeProb, IsFake: $isFake")
                
                val processingTimeMs = System.currentTimeMillis() - startTime
                
                Timber.d("✅ Raw audio analysis: ${if (isFake) "FAKE" else "REAL"} (${fakeProb * 100}% fake, ${realProb * 100}% real) - ${processingTimeMs}ms")
                
                AudioAnalysisResult(
                    isFake = isFake,
                    confidence = if (isFake) fakeProb else realProb,
                    fakeConfidence = fakeProb,
                    realConfidence = realProb,
                    processingTimeMs = processingTimeMs,
                    audioLengthMs = actualAudioLengthMs,
                    error = null
                )
                
            } catch (e: Exception) {
                Timber.e(e, "❌ Error analyzing raw audio")
                AudioAnalysisResult(
                    isFake = false,
                    confidence = 0f,
                    fakeConfidence = 0f,
                    realConfidence = 1f,
                    processingTimeMs = System.currentTimeMillis() - startTime,
                    audioLengthMs = audioLengthMs,
                    error = "Analysis failed: ${e.message}"
                )
            }
        }
    }
    
    /**
     * Analyze raw audio from ByteArray (e.g., from file or network)
     * Assumes 16-bit PCM format
     */
    suspend fun analyzeRawAudio(
        audioBytes: ByteArray,
        sampleRate: Int = SAMPLE_RATE,
        audioLengthMs: Long = -1L
    ): AudioAnalysisResult {
        return try {
            // Convert ByteArray to ShortArray (16-bit PCM)
            val audioData = ShortArray(audioBytes.size / 2)
            for (i in audioData.indices) {
                val low = audioBytes[i * 2].toInt() and 0xFF
                val high = audioBytes[i * 2 + 1].toInt() and 0xFF
                audioData[i] = ((high shl 8) or low).toShort()
            }
            
            analyzeRawAudio(audioData, sampleRate, audioLengthMs)
            
        } catch (e: Exception) {
            AudioAnalysisResult(
                isFake = false,
                confidence = 0f,
                fakeConfidence = 0f,
                realConfidence = 1f,
                processingTimeMs = 0L,
                audioLengthMs = audioLengthMs,
                error = "Failed to convert audio bytes: ${e.message}"
            )
        }
    }
    
    /**
     * Check if the service is ready for audio analysis
     */
    fun isReadyForAnalysis(): Boolean {
        return isModelLoaded.get() && deepfakeModel != null
    }
    
    fun getAudioSourceInfo(): String {
        return if (isRecording.get() && audioRecord != null) {
            val sourceName = when(currentAudioSource) {
                MediaRecorder.AudioSource.VOICE_CALL -> "Call audio"
                MediaRecorder.AudioSource.VOICE_DOWNLINK -> "Other caller"
                MediaRecorder.AudioSource.VOICE_UPLINK -> "Your voice"
                MediaRecorder.AudioSource.VOICE_COMMUNICATION -> "Microphone"
                else -> "Audio source"
            }
            "Recording: $sourceName"
        } else {
            "Not recording"
        }
    }
    
    /**
     * Check if device has root access for advanced call recording
     */
    private fun checkRootAccess(): Boolean {
        return try {
            val process = Runtime.getRuntime().exec("su")
            process.destroy()
            true
        } catch (e: Exception) {
            false
        }
    }
    
    /**
     * Get call recording capability info
     */
    fun getCallRecordingInfo(): String {
        return when {
            currentAudioSource == MediaRecorder.AudioSource.VOICE_DOWNLINK -> "Recording other caller"
            currentAudioSource == MediaRecorder.AudioSource.VOICE_CALL -> "Recording full call"
            else -> "Limited recording"
        }
    }
    

    
    /**
     * Check if speakerphone is currently active (modern API compatible)
     */
    private fun isSpeakerphoneActive(): Boolean {
        val audioManager = getSystemService(Context.AUDIO_SERVICE) as AudioManager
        
        return if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.S) {
            // Modern approach for Android 12+
            try {
                val communicationDevice = audioManager.communicationDevice
                communicationDevice?.type == AudioDeviceInfo.TYPE_BUILTIN_SPEAKER
            } catch (e: Exception) {
                // Fallback to deprecated method if modern approach fails
                @Suppress("DEPRECATION")
                audioManager.isSpeakerphoneOn
            }
        } else {
            // Use deprecated method for older Android versions
            @Suppress("DEPRECATION")
            audioManager.isSpeakerphoneOn
        }
    }
    
    /**
     * Enable speakerphone (modern API compatible)
     */
    private fun setSpeakerphoneOn(enabled: Boolean) {
        val audioManager = getSystemService(Context.AUDIO_SERVICE) as AudioManager
        
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.S) {
            // Modern approach for Android 12+
            try {
                if (enabled) {
                    // Store original device before changing
                    originalCommunicationDevice = audioManager.communicationDevice
                    
                    // Find built-in speaker
                    val devices = audioManager.availableCommunicationDevices
                    val speaker = devices.find { it.type == AudioDeviceInfo.TYPE_BUILTIN_SPEAKER }
                    
                    if (speaker != null) {
                        val success = audioManager.setCommunicationDevice(speaker)
                        if (success) {
                            Timber.d("✅ Modern speakerphone enabled successfully")
                        } else {
                            Timber.w("❌ Failed to set communication device to speaker")
                            // Fallback to deprecated method
                            @Suppress("DEPRECATION")
                            audioManager.isSpeakerphoneOn = enabled
                        }
                    } else {
                        Timber.w("❌ Built-in speaker not found in available devices")
                        // Fallback to deprecated method
                        @Suppress("DEPRECATION")
                        audioManager.isSpeakerphoneOn = enabled
                    }
                } else {
                    // Restore original device
                    val originalDevice = originalCommunicationDevice
                    if (originalDevice != null) {
                        audioManager.setCommunicationDevice(originalDevice)
                    } else {
                        audioManager.clearCommunicationDevice()
                    }
                }
            } catch (e: Exception) {
                Timber.w(e, "Modern speakerphone control failed, using fallback")
                // Fallback to deprecated method
                @Suppress("DEPRECATION")
                audioManager.isSpeakerphoneOn = enabled
            }
        } else {
            // Use deprecated method for older Android versions
            @Suppress("DEPRECATION")
            audioManager.isSpeakerphoneOn = enabled
        }
    }
    
    private fun showOverlay() {
        if (overlayView != null) return
        
        try {
            overlayView = OverlayView(this)
            
            val params = WindowManager.LayoutParams(
                WindowManager.LayoutParams.WRAP_CONTENT,
                WindowManager.LayoutParams.WRAP_CONTENT,
                if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.O) {
                    WindowManager.LayoutParams.TYPE_APPLICATION_OVERLAY
                } else {
                    @Suppress("DEPRECATION")
                    WindowManager.LayoutParams.TYPE_PHONE
                },
                WindowManager.LayoutParams.FLAG_NOT_FOCUSABLE or
                        WindowManager.LayoutParams.FLAG_NOT_TOUCH_MODAL,
                android.graphics.PixelFormat.TRANSLUCENT
            )
            
            windowManager?.addView(overlayView, params)
            Timber.d("Overlay view shown")
            
        } catch (e: Exception) {
            Timber.e(e, "Failed to show overlay")
        }
    }
    
    private fun hideOverlay() {
        overlayView?.let { view ->
            try {
                windowManager?.removeView(view)
                overlayView = null
                Timber.d("Overlay view hidden")
            } catch (e: Exception) {
                Timber.e(e, "Error hiding overlay")
            }
        }
    }
    
    private fun updateOverlay(result: DetectionResult) {
        overlayView?.updateDetectionResult(result)
    }
    
    private fun generateCallSummary() {
        if (detectionResults.isEmpty()) return
        
        val totalChunks = detectionResults.size
        val fakeChunks = detectionResults.count { it.isFake }
        val fakePercentage = (fakeChunks.toFloat() / totalChunks) * 100
        
        val avgConfidence = detectionResults.map { it.confidence }.average()
        
        Timber.i("Call Summary - Total chunks: $totalChunks, Fake: $fakeChunks (${fakePercentage.toInt()}%), Avg confidence: ${"%.2f".format(avgConfidence)}")
        
        // Could save to database or send to analytics here
        detectionResults.clear()
    }
    
    private fun initializeWindowManager() {
        windowManager = getSystemService(Context.WINDOW_SERVICE) as WindowManager
    }
    
    private fun createNotificationChannel() {
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.O) {
            val channel = NotificationChannel(
                CHANNEL_ID,
                getString(R.string.notification_channel_name),
                NotificationManager.IMPORTANCE_LOW
            ).apply {
                description = getString(R.string.notification_channel_description)
                setShowBadge(false)
                setSound(null, null)
            }
            
            val notificationManager = getSystemService(Context.NOTIFICATION_SERVICE) as NotificationManager
            notificationManager.createNotificationChannel(channel)
        }
    }
    
    private fun createNotification(contentText: String): Notification {
        val intent = Intent(this, MainActivity::class.java)
        val pendingIntent = PendingIntent.getActivity(
            this, 0, intent,
            PendingIntent.FLAG_UPDATE_CURRENT or PendingIntent.FLAG_IMMUTABLE
        )
        
        return NotificationCompat.Builder(this, CHANNEL_ID)
            .setContentTitle(getString(R.string.service_notification_title))
            .setContentText(contentText)
            .setSmallIcon(R.drawable.ic_launcher_foreground)
            .setContentIntent(pendingIntent)
            .setOngoing(true)
            .setSilent(true)
            .setCategory(NotificationCompat.CATEGORY_SERVICE)
            .build()
    }
    
    override fun onDestroy() {
        super.onDestroy()
        stopAudioMonitoring()
        hideOverlay()
        serviceScope.cancel()
        Timber.d("DeepfakeDetectionService destroyed")
    }
}
