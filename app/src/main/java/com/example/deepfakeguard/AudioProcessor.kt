/**
 * Multi-channel feature generator for deepfake detection model
 * Converts mono audio to 3-channel features: MelSpectrogram, MFCC, LFCC [3, 64, T]
 */
package com.example.deepfakeguard

import timber.log.Timber
import kotlin.math.*

/**
 * Result of multi-channel feature extraction
 */
data class MultiChannelFeaturesResult(
    val features: FloatArray,           // Flattened tensor data
    val shape: IntArray,                // [channels, features, time]
    val timeSteps: Int,                 // Number of time steps
    val error: String? = null          // Error message if any
)

class AudioProcessor {
    
    companion object {
        private const val TAG = "AudioProcessor"
        
        // Model input specs (must match training exactly)
        private const val NUM_FEATURE_BINS = 64    // n_mels, n_mfcc, n_lfcc
        private const val FFT_SIZE = 780           // n_fft (exact!)
        private const val HOP_LENGTH = 195         // hop_length
        private const val NUM_CHANNELS = 3         // MelSpec, MFCC, LFCC
        private const val N_FILTER = 64           // LFCC filter count
        
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
     * Generate 3-channel features for deepfake detection model
     * @param audioData 16-bit PCM audio (mono or stereo)
     * @param sampleRate Audio sample rate
     * @return Multi-channel features result containing [3, 64, T] tensor and metadata
     */
    fun generateMultiChannelFeatures(audioData: ShortArray, sampleRate: Int): MultiChannelFeaturesResult {
        try {
            // Convert to mono audio (average channels if stereo)
            val monoAudio = if (audioData.size % 2 == 0) {
                // Assume stereo, convert to mono
                FloatArray(audioData.size / 2) { i ->
                    (audioData[i * 2].toFloat() + audioData[i * 2 + 1].toFloat()) / (2f * Short.MAX_VALUE)
                }
            } else {
                // Already mono
                FloatArray(audioData.size) { i ->
                    audioData[i].toFloat() / Short.MAX_VALUE
                }
            }
            
            // Resample to 16kHz if needed (simplified - assumes already 16kHz for now)
            val processedAudio = if (sampleRate == 16000) {
                monoAudio
            } else {
                Timber.w("Sample rate $sampleRate != 16000. Resampling not implemented yet.")
                monoAudio
            }
            
            // Pad/trim to fixed length (6 seconds)
            val maxLen = 16000 * 6  // 6 seconds at 16kHz
            val waveform = if (processedAudio.size > maxLen) {
                processedAudio.sliceArray(0 until maxLen)
            } else {
                FloatArray(maxLen) { i ->
                    if (i < processedAudio.size) processedAudio[i] else 0f
                }
            }
            
            // Generate MelSpectrogram
            val melSpectrogram = computeMelSpectrogram(waveform)
            
            // Generate MFCC
            val mfcc = computeMFCC(waveform)
            
            // Generate LFCC  
            val lfcc = computeLFCC(waveform)
            
            // Align on time axis (find minimum frames)
            val minFrames = minOf(
                melSpectrogram[0].size,
                mfcc[0].size, 
                lfcc[0].size
            )
            
            // Prepare aligned features [3, 64, minFrames]
            val alignedMel = Array(NUM_FEATURE_BINS) { i ->
                melSpectrogram[i].sliceArray(0 until minFrames)
            }
            val alignedMfcc = Array(NUM_FEATURE_BINS) { i ->
                mfcc[i].sliceArray(0 until minFrames)
            }
            val alignedLfcc = Array(NUM_FEATURE_BINS) { i ->
                lfcc[i].sliceArray(0 until minFrames)
            }
            
            // Stack to [3, 64, minFrames] and flatten
            val totalSize = NUM_CHANNELS * NUM_FEATURE_BINS * minFrames
            val result = FloatArray(totalSize)
            
            // Channel 0: MelSpectrogram
            alignedMel.forEachIndexed { featureIdx, timeSteps ->
                timeSteps.forEachIndexed { timeIdx, value ->
                    val flatIdx = 0 * NUM_FEATURE_BINS * minFrames + featureIdx * minFrames + timeIdx
                    result[flatIdx] = value
                }
            }
            
            // Channel 1: MFCC
            alignedMfcc.forEachIndexed { featureIdx, timeSteps ->
                timeSteps.forEachIndexed { timeIdx, value ->
                    val flatIdx = 1 * NUM_FEATURE_BINS * minFrames + featureIdx * minFrames + timeIdx
                    result[flatIdx] = value
                }
            }
            
            // Channel 2: LFCC
            alignedLfcc.forEachIndexed { featureIdx, timeSteps ->
                timeSteps.forEachIndexed { timeIdx, value ->
                    val flatIdx = 2 * NUM_FEATURE_BINS * minFrames + featureIdx * minFrames + timeIdx
                    result[flatIdx] = value
                }
            }
            
            return MultiChannelFeaturesResult(
                features = result,
                shape = intArrayOf(NUM_CHANNELS, NUM_FEATURE_BINS, minFrames),
                timeSteps = minFrames
            )
            
        } catch (e: Exception) {
            Timber.e(e, "Multi-channel feature generation failed")
            return MultiChannelFeaturesResult(
                features = FloatArray(NUM_CHANNELS * NUM_FEATURE_BINS * 100) { 0f },
                shape = intArrayOf(NUM_CHANNELS, NUM_FEATURE_BINS, 100),
                timeSteps = 100,
                error = e.message
            )
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
    
    private fun computeMelSpectrogram(waveform: FloatArray): Array<FloatArray> {
        // Compute STFT
        val stftResult = computeSTFT(waveform)
        
        // Apply mel filter bank
        val melSpectrogram = applyMelFilterBank(stftResult)
        
        // Convert to dB scale
        return convertToDb(melSpectrogram)
    }
    
    private fun computeMFCC(waveform: FloatArray): Array<FloatArray> {
        // First get mel spectrogram
        val melSpectrogram = computeMelSpectrogram(waveform)
        
        // Apply DCT to mel spectrogram to get MFCC
        return applyDCT(melSpectrogram, NUM_FEATURE_BINS)
    }
    
    private fun computeLFCC(waveform: FloatArray): Array<FloatArray> {
        // Compute power spectrum from STFT
        val stftResult = computeSTFT(waveform)
        
        // Apply linear filter bank (instead of mel)
        val linearSpectrogram = applyLinearFilterBank(stftResult)
        
        // Convert to dB scale
        val dbSpectrogram = convertToDb(linearSpectrogram)
        
        // Apply DCT to get LFCC
        return applyDCT(dbSpectrogram, NUM_FEATURE_BINS)
    }
    
    private fun applyDCT(spectrogram: Array<FloatArray>, numCoeffs: Int): Array<FloatArray> {
        val numFrames = spectrogram[0].size
        val numBins = spectrogram.size
        val dctResult = Array(numCoeffs) { FloatArray(numFrames) }
        
        for (frameIdx in 0 until numFrames) {
            for (coeffIdx in 0 until numCoeffs) {
                var sum = 0.0
                for (binIdx in 0 until numBins) {
                    val cosArg = PI * coeffIdx * (2 * binIdx + 1) / (2 * numBins)
                    sum += spectrogram[binIdx][frameIdx] * cos(cosArg)
                }
                dctResult[coeffIdx][frameIdx] = sum.toFloat()
            }
        }
        
        return dctResult
    }
    
    private fun applyLinearFilterBank(stftMagnitude: Array<FloatArray>): Array<FloatArray> {
        val numFrames = stftMagnitude.size
        val numFreqBins = stftMagnitude[0].size
        val linearSpectrogram = Array(N_FILTER) { FloatArray(numFrames) }
        
        // Create linear filter bank (evenly spaced filters)
        val freqStep = numFreqBins.toFloat() / N_FILTER
        
        for (frameIdx in 0 until numFrames) {
            for (filterIdx in 0 until N_FILTER) {
                var filterSum = 0f
                val startBin = (filterIdx * freqStep).toInt()
                val endBin = ((filterIdx + 1) * freqStep).toInt().coerceAtMost(numFreqBins - 1)
                
                for (binIdx in startBin..endBin) {
                    filterSum += stftMagnitude[frameIdx][binIdx]
                }
                
                linearSpectrogram[filterIdx][frameIdx] = if (endBin > startBin) {
                    filterSum / (endBin - startBin + 1)
                } else {
                    filterSum
                }
            }
        }
        
        return linearSpectrogram
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
        val melSpectrogram = Array(NUM_FEATURE_BINS) { FloatArray(numFrames) }
        
        for (frameIdx in 0 until numFrames) {
            for (melIdx in 0 until NUM_FEATURE_BINS) {
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
        val numFeatures = melSpectrogram.size
        val dbSpectrogram = Array(numFeatures) { FloatArray(melSpectrogram[0].size) }
        
        // Find global max for TOP_DB clipping (like PyTorch AmplitudeToDB)
        var globalMax = Float.NEGATIVE_INFINITY
        
        // First pass: convert to dB and find max
        for (featureIdx in 0 until numFeatures) {
            for (timeIdx in melSpectrogram[featureIdx].indices) {
                val amplitude = max(melSpectrogram[featureIdx][timeIdx], 1e-10f)  // Avoid log(0)
                val db = 20f * log10(amplitude) - 20f * log10(REF_DB)  // Match PyTorch exactly
                dbSpectrogram[featureIdx][timeIdx] = db
                globalMax = max(globalMax, db)
            }
        }
        
        // Second pass: apply TOP_DB clipping relative to max (like PyTorch)
        val clipThreshold = globalMax - TOP_DB
        for (featureIdx in 0 until numFeatures) {
            for (timeIdx in melSpectrogram[featureIdx].indices) {
                dbSpectrogram[featureIdx][timeIdx] = max(dbSpectrogram[featureIdx][timeIdx], clipThreshold)
            }
        }
        
        return dbSpectrogram
    }
    
    /**
     * Backward compatibility method for existing code
     */
    fun generateMelSpectrogram(audioData: ShortArray, sampleRate: Int): FloatArray {
        val result = generateMultiChannelFeatures(audioData, sampleRate)
        return result.features
    }
    
    private fun createMelFilterBank(): Array<FloatArray> {
        val nyquist = 8000f  // Max frequency for 16kHz audio
        val melMin = hzToMel(MEL_MIN_FREQ)
        val melMax = hzToMel(min(MEL_MAX_FREQ, nyquist))
        
        // Create mel-spaced frequency points
        val melPoints = FloatArray(NUM_FEATURE_BINS + 2) { i ->
            melMin + (melMax - melMin) * i / (NUM_FEATURE_BINS + 1)
        }
        
        // Convert back to Hz
        val hzPoints = melPoints.map { melToHz(it) }
        
        // Convert to FFT bin indices
        val binIndices = hzPoints.map { hz ->
            ((FFT_SIZE / 2 + 1) * hz / nyquist).toInt().coerceAtMost(FFT_SIZE / 2)
        }
        
        // Create triangular filter bank for current FFT size
        val filterBank = Array(NUM_FEATURE_BINS) { FloatArray(FFT_SIZE / 2 + 1) }
        
        for (melIdx in 0 until NUM_FEATURE_BINS) {
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
