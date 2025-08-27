/**
 * Call overlay for real-time deepfake detection results
 * Shows visual feedback during phone calls
 */
package com.example.deepfakeguard

import android.animation.ValueAnimator
import android.content.Context
import android.graphics.*
import android.util.AttributeSet
import android.view.View
import android.view.animation.AccelerateDecelerateInterpolator
import androidx.core.content.ContextCompat
import timber.log.Timber
import kotlin.math.*

class OverlayView @JvmOverloads constructor(
    context: Context,
    attrs: AttributeSet? = null,
    defStyleAttr: Int = 0
) : View(context, attrs, defStyleAttr) {
    
    companion object {
        private const val DETECTION_HISTORY_SIZE = 20
        private const val ANIMATION_DURATION = 300L
        private const val CONFIDENCE_THRESHOLD = 0.7f
    }
    
    // UI colors
    private val colorGreen = Color.rgb(46, 204, 113)
    private val colorRed = Color.rgb(231, 76, 60)
    private val colorYellow = Color.rgb(241, 196, 15)
    private val colorGray = Color.rgb(149, 165, 166)
    private val colorBackground = Color.argb(200, 0, 0, 0)
    
    // Paint setup
    private val backgroundPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        color = colorBackground
        style = Paint.Style.FILL
    }
    
    private val statusPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        textAlign = Paint.Align.CENTER
        textSize = 48f
        typeface = Typeface.DEFAULT_BOLD
    }
    
    private val detailPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        textAlign = Paint.Align.CENTER
        textSize = 32f
        typeface = Typeface.DEFAULT
    }
    
    private val chartPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        style = Paint.Style.STROKE
        strokeWidth = 6f
    }
    
    private val fillPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        style = Paint.Style.FILL
    }
    
    // State
    private var currentResult: DeepfakeDetectionService.DetectionResult? = null
    private val detectionHistory = mutableListOf<Float>()
    private var isAnalyzing = true
    
    // Animation & layout
    private var pulseAnimator: ValueAnimator? = null
    private var pulseScale = 1f
    private val cornerRadius = 24f
    private val padding = 32f
    private val chartHeight = 120f
    
    init {
        // Setup initial state
        layoutParams = android.view.ViewGroup.LayoutParams(
            android.view.ViewGroup.LayoutParams.WRAP_CONTENT,
            android.view.ViewGroup.LayoutParams.WRAP_CONTENT
        )
        startPulseAnimation()
    }
    
    fun updateDetectionResult(result: DeepfakeDetectionService.DetectionResult) {
        currentResult = result
        isAnalyzing = false
        
        // Update history and animation
        detectionHistory.add(result.confidence)
        if (detectionHistory.size > DETECTION_HISTORY_SIZE) {
            detectionHistory.removeAt(0)
        }
        updatePulseAnimation(result.isFake, result.confidence)
        
        invalidate()
        
        Timber.d("Overlay: Fake=${result.isFake}, Conf=${result.confidence}")
    }
    
    override fun onMeasure(widthMeasureSpec: Int, heightMeasureSpec: Int) {
        val width = 400
        val height = 300
        setMeasuredDimension(width, height)
    }
    
    override fun onDraw(canvas: Canvas) {
        super.onDraw(canvas)
        
        val width = measuredWidth.toFloat()
        val height = measuredHeight.toFloat()
        
        // Background
        canvas.drawRoundRect(RectF(0f, 0f, width, height), cornerRadius, cornerRadius, backgroundPaint)
        
        // Content layers
        drawMainStatus(canvas, width, height)
        drawConfidenceChart(canvas, width, height)
        drawDetailInfo(canvas, width, height)
    }
    
    private fun drawMainStatus(canvas: Canvas, width: Float, height: Float) {
        val centerX = width / 2f
        val statusY = padding + 60f
        
        val (statusText, statusColor) = when {
            isAnalyzing -> "🔍 ANALYZING" to colorYellow
            currentResult == null -> "⏳ WAITING" to colorGray
            currentResult!!.isFake && currentResult!!.confidence > CONFIDENCE_THRESHOLD -> 
                "⚠️ DEEPFAKE" to colorRed
            currentResult!!.isFake -> "⚠️ SUSPICIOUS" to colorYellow
            else -> "✅ AUTHENTIC" to colorGreen
        }
        
        // Apply pulse effect
        statusPaint.textSize = 48f * pulseScale
        statusPaint.color = statusColor
        canvas.drawText(statusText, centerX, statusY, statusPaint)
        statusPaint.textSize = 48f // Reset
    }
    
    private fun drawConfidenceChart(canvas: Canvas, width: Float, height: Float) {
        if (detectionHistory.isEmpty()) return
        
        val chartX = padding
        val chartY = height - chartHeight - padding * 2
        val chartWidth = width - padding * 2
        
        // Chart background
        fillPaint.color = Color.argb(100, 255, 255, 255)
        canvas.drawRoundRect(RectF(chartX, chartY, chartX + chartWidth, chartY + chartHeight), 12f, 12f, fillPaint)
        
        // Confidence trend line
        if (detectionHistory.size > 1) {
            val path = Path()
            val stepX = chartWidth / (DETECTION_HISTORY_SIZE - 1)
            
            for (i in detectionHistory.indices) {
                val x = chartX + i * stepX * (detectionHistory.size - 1) / (DETECTION_HISTORY_SIZE - 1)
                val y = chartY + chartHeight - (detectionHistory[i] * chartHeight)
                
                if (i == 0) {
                    path.moveTo(x, y)
                } else {
                    path.lineTo(x, y)
                }
            }
            
            // Color by average confidence
            chartPaint.color = when (detectionHistory.average()) {
                in 0.7..1.0 -> colorRed
                in 0.3..0.7 -> colorYellow
                else -> colorGreen
            }
            
            canvas.drawPath(path, chartPaint)
        }
        
        // Threshold line (0.5)
        chartPaint.color = Color.argb(150, 255, 255, 255)
        chartPaint.pathEffect = DashPathEffect(floatArrayOf(10f, 10f), 0f)
        val thresholdY = chartY + chartHeight - (0.5f * chartHeight)
        canvas.drawLine(chartX, thresholdY, chartX + chartWidth, thresholdY, chartPaint)
        chartPaint.pathEffect = null
    }
    
    private fun drawDetailInfo(canvas: Canvas, width: Float, height: Float) {
        val centerX = width / 2f
        val detailY = height - padding - 20f
        
        val detailText = when {
            currentResult == null -> "Waiting for audio data..."
            else -> {
                val confidence = (currentResult!!.confidence * 100).toInt()
                val samples = detectionHistory.size
                "Confidence: $confidence% • Samples: $samples"
            }
        }
        
        detailPaint.color = Color.argb(200, 255, 255, 255)
        canvas.drawText(detailText, centerX, detailY, detailPaint)
    }
    
    private fun startPulseAnimation() {
        pulseAnimator = ValueAnimator.ofFloat(1f, 1.2f, 1f).apply {
            duration = 1500L
            repeatCount = ValueAnimator.INFINITE
            interpolator = AccelerateDecelerateInterpolator()
            addUpdateListener { animation ->
                pulseScale = animation.animatedValue as Float
                invalidate()
            }
        }
        pulseAnimator?.start()
    }
    
    private fun updatePulseAnimation(isFake: Boolean, confidence: Float) {
        pulseAnimator?.cancel()
        
        val (scale, duration) = when {
            isFake && confidence > CONFIDENCE_THRESHOLD -> 1.4f to 400L  // Urgent
            isFake -> 1.2f to 800L  // Suspicious
            else -> 1.1f to 2000L   // Authentic
        }
        
        pulseAnimator = ValueAnimator.ofFloat(1f, scale, 1f).apply {
            this.duration = duration
            repeatCount = ValueAnimator.INFINITE
            interpolator = AccelerateDecelerateInterpolator()
            addUpdateListener { animation ->
                pulseScale = animation.animatedValue as Float
                invalidate()
            }
        }
        
        pulseAnimator?.start()
    }
    
    override fun onDetachedFromWindow() {
        super.onDetachedFromWindow()
        pulseAnimator?.cancel()
    }
}
