/**
 * Mel spectrogram generator for CRNN deepfake model
 * Converts stereo audio to model-compatible mel spectrograms [2, 64, 300]
 */
package com.example.deepfakeguard

import timber.log.Timber
import kotlin.math.*

class AudioProcessor {
    
    companion object {
        private const val TAG = "AudioProcessor"
        
        // Model input specs (must match training exactly)
        private const val NUM_MEL_BINS = 64        // n_mels
        private const val FFT_SIZE = 780           // n_fft (exact!)
        private const val HOP_LENGTH = 195         // hop_length
        private const val NUM_CHANNELS = 2         // Stereo
        private const val TARGET_TIME_STEPS = 300  // Time dimension
        
        // ImageNet normalization (training match)
        private val IMAGENET_MEAN = floatArrayOf(0.485f, 0.456f)
        private val IMAGENET_STD = floatArrayOf(0.229f, 0.224f)
        
        // Mel filter parameters
        private const val MEL_MIN_FREQ = 80f
        private const val MEL_MAX_FREQ = 8000f
        
        // dB conversion
        private const val TOP_DB = 80f
        private const val REF_DB = 1.0f
        
        // Hamming window for STFT
        private fun hammingWindow(size: Int): FloatArray = FloatArray(size) { i ->
            (0.54 - 0.46 * cos(2.0 * PI * i / (size - 1))).toFloat()
        }
        
        // Frequency conversions
        private fun hzToMel(hz: Float): Float = 2595f * log10(1f + hz / 700f)
        private fun melToHz(mel: Float): Float = 700f * (10f.pow(mel / 2595f) - 1f)
        
        // DFT for FFT size 780
        private fun computeDFT(signal: FloatArray): Array<Complex> {
            val N = signal.size
            val result = Array(N) { Complex(0.0, 0.0) }
            
            // Pre-compute trig tables
            val cosTable = DoubleArray(N) { k -> cos(-2.0 * PI * k / N) }
            val sinTable = DoubleArray(N) { k -> sin(-2.0 * PI * k / N) }
            
            // First half + DC only
            val halfN = N / 2 + 1
            
            for (k in 0 until halfN) {
                var sumReal = 0.0
                var sumImag = 0.0
                
                for (n in 0 until N) {
                    val tableIndex = (k * n) % N
                    sumReal += signal[n] * cosTable[tableIndex]
                    sumImag += signal[n] * sinTable[tableIndex]
                }
                
                result[k] = Complex(sumReal, sumImag)
            }
            
            return result
        }
        
        // Complex number
        data class Complex(val real: Double, val imag: Double) {
            fun abs(): Double = sqrt(real * real + imag * imag)
        }
    }
    
    private val hammingWindow = hammingWindow(FFT_SIZE)
    private val melFilterBank = createMelFilterBank()
    
    /**
     * Generate mel spectrogram for CRNN model
     * @param audioData 16-bit PCM stereo (interleaved)
     * @param sampleRate Audio sample rate
     * @return Flattened tensor [2, 64, 300] -> [38400]
     */
    fun generateMelSpectrogram(audioData: ShortArray, sampleRate: Int): FloatArray {
        try {
            // Split stereo and compute mel spectrograms
            val (leftChannel, rightChannel) = separateStereoChannels(audioData)
            val leftMelSpec = computeChannelMelSpectrogram(leftChannel, sampleRate)
            val rightMelSpec = computeChannelMelSpectrogram(rightChannel, sampleRate)
            
            // Prepare output tensor [2, 64, 300]
            val result = FloatArray(2 * NUM_MEL_BINS * TARGET_TIME_STEPS)
            val leftPadded = Array(NUM_MEL_BINS) { FloatArray(TARGET_TIME_STEPS) }
            val rightPadded = Array(NUM_MEL_BINS) { FloatArray(TARGET_TIME_STEPS) }
            
            // Pad/trim to exact target size
            for (melIdx in 0 until NUM_MEL_BINS) {
                for (timeIdx in 0 until TARGET_TIME_STEPS) {
                    leftPadded[melIdx][timeIdx] = if (timeIdx < leftMelSpec[melIdx].size) {
                        leftMelSpec[melIdx][timeIdx]
                    } else {
                        leftMelSpec[melIdx].lastOrNull() ?: 0f
                    }
                    
                    rightPadded[melIdx][timeIdx] = if (timeIdx < rightMelSpec[melIdx].size) {
                        rightMelSpec[melIdx][timeIdx]
                    } else {
                        rightMelSpec[melIdx].lastOrNull() ?: 0f
                    }
                }
            }
            

            
            // Channel 0 (left)
            leftPadded.forEachIndexed { melIdx, timeSteps ->
                timeSteps.forEachIndexed { timeIdx, value ->
                    val flatIdx = 0 * NUM_MEL_BINS * TARGET_TIME_STEPS + melIdx * TARGET_TIME_STEPS + timeIdx
                    result[flatIdx] = value
                }
            }
            
            // Channel 1 (right)
            rightPadded.forEachIndexed { melIdx, timeSteps ->
                timeSteps.forEachIndexed { timeIdx, value ->
                    val flatIdx = 1 * NUM_MEL_BINS * TARGET_TIME_STEPS + melIdx * TARGET_TIME_STEPS + timeIdx
                    result[flatIdx] = value
                }
            }
            
            // Apply ImageNet normalization per channel
            for (ch in 0 until NUM_CHANNELS) {
                val channelOffset = ch * NUM_MEL_BINS * TARGET_TIME_STEPS
                for (i in 0 until NUM_MEL_BINS * TARGET_TIME_STEPS) {
                    val idx = channelOffset + i
                    if (idx < result.size) {
                        result[idx] = (result[idx] - IMAGENET_MEAN[ch]) / IMAGENET_STD[ch]
                    }
                }
            }
            

            
            return result
            
        } catch (e: Exception) {
            Timber.e(e, "Mel spectrogram generation failed")
            return FloatArray(2 * NUM_MEL_BINS * TARGET_TIME_STEPS) { 0f }
        }
    }
    
    private fun separateStereoChannels(interleavedData: ShortArray): Pair<FloatArray, FloatArray> {
        val numSamples = interleavedData.size / 2
        val leftChannel = FloatArray(numSamples)
        val rightChannel = FloatArray(numSamples)
        
        for (i in 0 until numSamples) {
            leftChannel[i] = interleavedData[i * 2].toFloat() / Short.MAX_VALUE
            rightChannel[i] = interleavedData[i * 2 + 1].toFloat() / Short.MAX_VALUE
        }
        
        return Pair(leftChannel, rightChannel)
    }
    
    private fun computeChannelMelSpectrogram(audioChannel: FloatArray, sampleRate: Int): Array<FloatArray> {
        // Compute STFT
        val stftResult = computeSTFT(audioChannel)
        
        // Apply mel filter bank
        val melSpectrogram = applyMelFilterBank(stftResult)
        
        // Convert to dB scale
        return convertToDb(melSpectrogram)
    }
    
    private fun computeSTFT(audio: FloatArray): Array<FloatArray> {
        val numFrames = (audio.size - FFT_SIZE) / HOP_LENGTH + 1
        val numFreqBins = FFT_SIZE / 2 + 1
        val stftMagnitude = Array(numFrames) { FloatArray(numFreqBins) }
        
        for (frameIdx in 0 until numFrames) {
            
            val startIdx = frameIdx * HOP_LENGTH
            val frame = FloatArray(FFT_SIZE)
            
            // Extract frame and apply Hamming window
            for (i in 0 until FFT_SIZE) {
                val audioIdx = startIdx + i
                frame[i] = if (audioIdx < audio.size) {
                    audio[audioIdx] * hammingWindow[i]
                } else {
                    0f
                }
            }
            
            // Compute DFT using custom implementation
            val dftResult = Companion.computeDFT(frame)
            
            // Compute magnitude spectrum (only first half + DC)
            for (i in 0 until numFreqBins) {
                stftMagnitude[frameIdx][i] = dftResult[i].abs().toFloat()
            }
        }
        
        return stftMagnitude
    }
    
    private fun applyMelFilterBank(stftMagnitude: Array<FloatArray>): Array<FloatArray> {
        val numFrames = stftMagnitude.size
        val melSpectrogram = Array(NUM_MEL_BINS) { FloatArray(numFrames) }
        
        for (frameIdx in 0 until numFrames) {
            for (melIdx in 0 until NUM_MEL_BINS) {
                var melEnergy = 0f
                for (freqIdx in stftMagnitude[frameIdx].indices) {
                    melEnergy += stftMagnitude[frameIdx][freqIdx] * melFilterBank[melIdx][freqIdx]
                }
                melSpectrogram[melIdx][frameIdx] = melEnergy
            }
        }
        
        return melSpectrogram
    }
    
    private fun convertToDb(melSpectrogram: Array<FloatArray>): Array<FloatArray> {
        val dbSpectrogram = Array(NUM_MEL_BINS) { FloatArray(melSpectrogram[0].size) }
        
        // Find global max for TOP_DB clipping (like PyTorch AmplitudeToDB)
        var globalMax = Float.NEGATIVE_INFINITY
        
        // First pass: convert to dB and find max
        for (melIdx in 0 until NUM_MEL_BINS) {
            for (timeIdx in melSpectrogram[melIdx].indices) {
                val amplitude = max(melSpectrogram[melIdx][timeIdx], 1e-10f)  // Avoid log(0)
                val db = 20f * log10(amplitude) - 20f * log10(REF_DB)  // Match PyTorch exactly
                dbSpectrogram[melIdx][timeIdx] = db
                globalMax = max(globalMax, db)
            }
        }
        
        // Second pass: apply TOP_DB clipping relative to max (like PyTorch)
        val clipThreshold = globalMax - TOP_DB
        for (melIdx in 0 until NUM_MEL_BINS) {
            for (timeIdx in melSpectrogram[melIdx].indices) {
                dbSpectrogram[melIdx][timeIdx] = max(dbSpectrogram[melIdx][timeIdx], clipThreshold)
            }
        }
        

        
        return dbSpectrogram
    }
    
    private fun createMelFilterBank(): Array<FloatArray> {
        val nyquist = 8000f  // Max frequency for 16kHz audio
        val melMin = hzToMel(MEL_MIN_FREQ)
        val melMax = hzToMel(min(MEL_MAX_FREQ, nyquist))
        
        // Create mel-spaced frequency points
        val melPoints = FloatArray(NUM_MEL_BINS + 2) { i ->
            melMin + (melMax - melMin) * i / (NUM_MEL_BINS + 1)
        }
        
        // Convert back to Hz
        val hzPoints = melPoints.map { melToHz(it) }
        
        // Convert to FFT bin indices
        val binIndices = hzPoints.map { hz ->
            ((FFT_SIZE / 2 + 1) * hz / nyquist).toInt().coerceAtMost(FFT_SIZE / 2)
        }
        
        // Create triangular filter bank for current FFT size
        val filterBank = Array(NUM_MEL_BINS) { FloatArray(FFT_SIZE / 2 + 1) }
        
        for (melIdx in 0 until NUM_MEL_BINS) {
            val leftBin = binIndices[melIdx]
            val centerBin = binIndices[melIdx + 1]
            val rightBin = binIndices[melIdx + 2]
            
            // Create triangular filter
            for (binIdx in leftBin..rightBin) {
                if (binIdx < FFT_SIZE / 2 + 1) {
                    when {
                        binIdx < centerBin && centerBin > leftBin -> {
                            filterBank[melIdx][binIdx] = (binIdx - leftBin).toFloat() / (centerBin - leftBin)
                        }
                        binIdx >= centerBin && rightBin > centerBin -> {
                            filterBank[melIdx][binIdx] = (rightBin - binIdx).toFloat() / (rightBin - centerBin)
                        }
                    }
                }
            }
        }
        
        return filterBank
    }
}
