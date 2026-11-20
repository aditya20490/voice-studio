package com.voicestudio.app

import android.Manifest
import android.content.pm.PackageManager
import android.os.Bundle
import android.webkit.MimeTypeMap
import androidx.activity.ComponentActivity
import androidx.activity.compose.rememberLauncherForActivityResult
import androidx.activity.compose.setContent
import androidx.activity.result.contract.ActivityResultContracts
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.foundation.lazy.items
import androidx.compose.material3.Button
import androidx.compose.material3.Card
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.NavigationBar
import androidx.compose.material3.NavigationBarItem
import androidx.compose.material3.OutlinedButton
import androidx.compose.material3.OutlinedTextField
import androidx.compose.material3.Scaffold
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.DisposableEffect
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableIntStateOf
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.Modifier
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.unit.dp
import androidx.core.content.ContextCompat
import kotlinx.coroutines.delay
import java.io.File

class MainActivity : ComponentActivity() {
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContent { MaterialTheme { App() } }
    }
}

@Composable
fun App() {
    val ctx = LocalContext.current
    val store = remember { VoiceStore(ctx) }
    var voices by remember { mutableStateOf(store.list()) }
    var tab by remember { mutableIntStateOf(0) }

    Scaffold(
        bottomBar = {
            NavigationBar {
                listOf("My Voices", "Create Voice", "Speak").forEachIndexed { i, title ->
                    NavigationBarItem(
                        selected = tab == i,
                        onClick = { tab = i },
                        icon = {},
                        label = { Text(title) }
                    )
                }
            }
        }
    ) { pad ->
        Box(Modifier.padding(pad).fillMaxSize()) {
            when (tab) {
                0 -> VoicesTab(voices) { id -> store.delete(id); voices = store.list() }
                1 -> CreateTab(store) { voices = store.list(); tab = 0 }
                else -> SpeakTab(voices)
            }
        }
    }
}

@Composable
fun VoicesTab(voices: List<Voice>, onDelete: (String) -> Unit) {
    if (voices.isEmpty()) {
        Box(Modifier.fillMaxSize().padding(24.dp)) {
            Text("No voices yet. Open the Create Voice tab to record or import your voice.")
        }
        return
    }
    LazyColumn(Modifier.fillMaxSize().padding(16.dp), verticalArrangement = Arrangement.spacedBy(12.dp)) {
        items(voices, key = { it.id }) { v ->
            Card(Modifier.fillMaxWidth()) {
                Column(Modifier.padding(16.dp)) {
                    Text(v.name, style = MaterialTheme.typography.titleMedium)
                    Text("${v.clips.size} recording(s) saved")
                    OutlinedButton(onClick = { onDelete(v.id) }) { Text("Delete") }
                }
            }
        }
    }
}

@Composable
fun CreateTab(store: VoiceStore, onSaved: () -> Unit) {
    val ctx = LocalContext.current
    val pendingDir = remember { File(ctx.cacheDir, "pending").apply { mkdirs() } }
    val recorder = remember { WavRecorder() }
    var name by remember { mutableStateOf("") }
    var clips by remember { mutableStateOf(pendingDir.listFiles()?.sortedBy { it.name } ?: emptyList()) }
    var recording by remember { mutableStateOf(false) }
    var secs by remember { mutableIntStateOf(0) }

    fun refresh() {
        clips = pendingDir.listFiles()?.sortedBy { it.name } ?: emptyList()
    }

    fun startRecording() {
        recorder.start(File(pendingDir, "rec_${System.currentTimeMillis()}.wav"))
        recording = true
    }

    fun stopRecording() {
        recorder.stop()
        recording = false
        refresh()
    }

    val permission = rememberLauncherForActivityResult(ActivityResultContracts.RequestPermission()) { ok ->
        if (ok) startRecording()
    }

    val picker = rememberLauncherForActivityResult(ActivityResultContracts.GetContent()) { uri ->
        if (uri != null) {
            val ext = MimeTypeMap.getSingleton()
                .getExtensionFromMimeType(ctx.contentResolver.getType(uri)) ?: "audio"
            val out = File(pendingDir, "import_${System.currentTimeMillis()}.$ext")
            ctx.contentResolver.openInputStream(uri)?.use { input ->
                out.outputStream().use { input.copyTo(it) }
            }
            refresh()
        }
    }

    LaunchedEffect(recording) {
        while (recording) {
            secs = recorder.seconds
            delay(300)
        }
    }

    DisposableEffect(Unit) {
        onDispose { if (recorder.isRecording) recorder.stop() }
    }

    LazyColumn(Modifier.fillMaxSize().padding(16.dp), verticalArrangement = Arrangement.spacedBy(12.dp)) {
        item {
            Text("Create your voice", style = MaterialTheme.typography.headlineSmall)
            Text(
                "Tip: record 30 to 60 seconds in a quiet room. Speak some Hindi and some English, " +
                    "clearly and naturally. Use only your own voice, or a voice you have permission to use."
            )
        }
        item {
            OutlinedTextField(
                value = name,
                onValueChange = { name = it },
                label = { Text("Voice name") },
                modifier = Modifier.fillMaxWidth()
            )
        }
        item {
            Row(horizontalArrangement = Arrangement.spacedBy(12.dp)) {
                Button(onClick = {
                    if (recording) {
                        stopRecording()
                    } else if (ContextCompat.checkSelfPermission(ctx, Manifest.permission.RECORD_AUDIO)
                        == PackageManager.PERMISSION_GRANTED
                    ) {
                        startRecording()
                    } else {
                        permission.launch(Manifest.permission.RECORD_AUDIO)
                    }
                }) { Text(if (recording) "Stop ($secs s)" else "Record") }
                OutlinedButton(onClick = { picker.launch("audio/*") }, enabled = !recording) {
                    Text("Import audio file")
                }
            }
        }
        items(clips, key = { it.name }) { f ->
            Card(Modifier.fillMaxWidth()) {
                Row(
                    Modifier.padding(12.dp).fillMaxWidth(),
                    horizontalArrangement = Arrangement.SpaceBetween
                ) {
                    val label = if (f.extension == "wav" && f.name.startsWith("rec_"))
                        "Recording, about ${(f.length() - 44) / (WavRecorder.SAMPLE_RATE * 2)} s"
                    else "Imported file, ${f.length() / 1024} KB"
                    Text(label)
                    OutlinedButton(onClick = { f.delete(); refresh() }) { Text("Remove") }
                }
            }
        }
        item {
            Button(
                onClick = {
                    store.create(name, clips)
                    name = ""
                    refresh()
                    onSaved()
                },
                enabled = name.isNotBlank() && clips.isNotEmpty() && !recording,
                modifier = Modifier.fillMaxWidth()
            ) { Text("Save voice") }
        }
    }
}
