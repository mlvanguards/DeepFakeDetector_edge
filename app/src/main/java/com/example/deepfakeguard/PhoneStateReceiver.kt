/**
 * BroadcastReceiver for phone call detection
 * Auto-starts/stops deepfake detection service based on call state
 */
package com.example.deepfakeguard

import android.content.BroadcastReceiver
import android.content.Context
import android.content.Intent
import android.telephony.TelephonyManager

import timber.log.Timber

class PhoneStateReceiver : BroadcastReceiver() {
    
    companion object {
        private var lastState = TelephonyManager.CALL_STATE_IDLE
        private var isIncoming = false
        private var callerNumber: String? = null
    }
    
    override fun onReceive(context: Context, intent: Intent) {
        try {
            when (intent.action) {
                TelephonyManager.ACTION_PHONE_STATE_CHANGED -> handlePhoneStateChange(context, intent)
                @Suppress("DEPRECATION")
                Intent.ACTION_NEW_OUTGOING_CALL -> handleOutgoingCall(context, intent)  // Deprecated but functional
            }
        } catch (e: Exception) {
            Timber.e(e, "PhoneStateReceiver error")
        }
    }
    
    private fun handlePhoneStateChange(context: Context, intent: Intent) {
        val state = intent.getStringExtra(TelephonyManager.EXTRA_STATE)
        @Suppress("DEPRECATION")
        val incomingNumber = intent.getStringExtra(TelephonyManager.EXTRA_INCOMING_NUMBER)
        
        val currentState = when (state) {
            TelephonyManager.EXTRA_STATE_IDLE -> TelephonyManager.CALL_STATE_IDLE
            TelephonyManager.EXTRA_STATE_RINGING -> TelephonyManager.CALL_STATE_RINGING
            TelephonyManager.EXTRA_STATE_OFFHOOK -> TelephonyManager.CALL_STATE_OFFHOOK
            else -> TelephonyManager.CALL_STATE_IDLE
        }
        
        Timber.d("State: $state (curr: $currentState, last: $lastState)")
        
        when {
            // Incoming call ringing
            currentState == TelephonyManager.CALL_STATE_RINGING && lastState != TelephonyManager.CALL_STATE_RINGING -> {
                isIncoming = true
                callerNumber = incomingNumber
                Timber.i("Incoming: $incomingNumber")
                prepareForCall(context, incomingNumber, true)
            }
            
            // Call connected
            currentState == TelephonyManager.CALL_STATE_OFFHOOK && lastState != TelephonyManager.CALL_STATE_OFFHOOK -> {
                Timber.i("Connected: incoming=$isIncoming, number=$callerNumber")
                startDeepfakeDetection(context, callerNumber, isIncoming)
            }
            
            // Call ended
            currentState == TelephonyManager.CALL_STATE_IDLE && lastState != TelephonyManager.CALL_STATE_IDLE -> {
                Timber.i("Call ended")
                stopDeepfakeDetection(context)
                resetCallState()
            }
        }
        
        lastState = currentState
    }
    
    private fun handleOutgoingCall(context: Context, intent: Intent) {
        @Suppress("DEPRECATION")
        val outgoingNumber = intent.getStringExtra(Intent.EXTRA_PHONE_NUMBER)
        Timber.i("Outgoing: $outgoingNumber")
        
        isIncoming = false
        callerNumber = outgoingNumber
        prepareForCall(context, outgoingNumber, false)
    }
    
    private fun prepareForCall(context: Context, phoneNumber: String?, incoming: Boolean) {
        Timber.d("Preparing: $phoneNumber, incoming=$incoming")
        
        // Pre-load ML model
        val prepareIntent = Intent(context, DeepfakeDetectionService::class.java).apply {
            action = DeepfakeDetectionService.ACTION_PREPARE
            putExtra(DeepfakeDetectionService.EXTRA_PHONE_NUMBER, phoneNumber)
            putExtra(DeepfakeDetectionService.EXTRA_IS_INCOMING, incoming)
        }
        
        try {
            context.startForegroundService(prepareIntent)
        } catch (e: Exception) {
            Timber.e(e, "Failed to start preparation service")
        }
    }
    
    private fun startDeepfakeDetection(context: Context, phoneNumber: String?, incoming: Boolean) {
        Timber.i("Starting detection service")
        
        val serviceIntent = Intent(context, DeepfakeDetectionService::class.java).apply {
            action = DeepfakeDetectionService.ACTION_START_DETECTION
            putExtra(DeepfakeDetectionService.EXTRA_PHONE_NUMBER, phoneNumber)
            putExtra(DeepfakeDetectionService.EXTRA_IS_INCOMING, incoming)
        }
        
        try {
            context.startForegroundService(serviceIntent)
        } catch (e: Exception) {
            Timber.e(e, "Failed to start deepfake detection service")
        }
    }
    
    private fun stopDeepfakeDetection(context: Context) {
        Timber.i("Stopping detection service")
        
        val serviceIntent = Intent(context, DeepfakeDetectionService::class.java).apply {
            action = DeepfakeDetectionService.ACTION_STOP_DETECTION
        }
        
        try {
            context.startService(serviceIntent)
        } catch (e: Exception) {
            Timber.e(e, "Failed to stop deepfake detection service")
        }
    }
    
    private fun resetCallState() {
        isIncoming = false
        callerNumber = null
    }
}
