package com.voicestudio.app

import android.content.Context
import java.io.File
import java.util.UUID

data class Voice(val id: String, val name: String, val clips: List<File>)

/** Saves voices as folders inside the app's private storage, so they survive restarts and updates. */
class VoiceStore(context: Context) {
    private val root = File(context.filesDir, "voices").apply { mkdirs() }

    fun list(): List<Voice> =
        (root.listFiles() ?: emptyArray())
            .filter { it.isDirectory }
            .sortedBy { it.lastModified() }
            .map { dir ->
                val name = File(dir, "name.txt").takeIf { it.exists() }?.readText() ?: dir.name
                val clips = (dir.listFiles() ?: emptyArray()).filter { it.name != "name.txt" }.sortedBy { it.name }
                Voice(dir.name, name, clips)
            }

    fun create(name: String, clips: List<File>) {
        val dir = File(root, UUID.randomUUID().toString()).apply { mkdirs() }
        File(dir, "name.txt").writeText(name.trim())
        clips.forEachIndexed { i, f ->
            val ext = f.extension.ifEmpty { "audio" }
            f.copyTo(File(dir, "clip_%02d.%s".format(i + 1, ext)), overwrite = true)
            f.delete()
        }
    }

    fun delete(id: String) {
        File(root, id).deleteRecursively()
    }
}
