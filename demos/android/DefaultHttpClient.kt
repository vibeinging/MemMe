package com.memme.demo

import java.io.OutputStreamWriter
import java.net.HttpURLConnection
import java.net.URL
import uniffi.memme_ffi.HttpClient
import uniffi.memme_ffi.MemmeException

/**
 * HttpURLConnection-based HttpClient that satisfies the UniFFI callback interface.
 *
 * For production use, replace with OkHttp for better performance:
 *
 *   class OkHttpClient(private val client: okhttp3.OkHttpClient = okhttp3.OkHttpClient()) : HttpClient {
 *       override fun post(url: String, headers: List<String>, body: String): String {
 *           val mediaType = "application/json".toMediaType()
 *           val requestBody = body.toRequestBody(mediaType)
 *           val builder = Request.Builder().url(url).post(requestBody)
 *           var i = 0
 *           while (i + 1 < headers.size) {
 *               builder.addHeader(headers[i], headers[i + 1])
 *               i += 2
 *           }
 *           client.newCall(builder.build()).execute().use { response ->
 *               return response.body?.string() ?: throw MemmeException.Runtime("Empty response")
 *           }
 *       }
 *   }
 */
class DefaultHttpClient : HttpClient {
    override fun post(url: String, headers: List<String>, body: String): String {
        val connection = URL(url).openConnection() as HttpURLConnection
        try {
            connection.requestMethod = "POST"
            connection.doOutput = true
            connection.connectTimeout = 30_000
            connection.readTimeout = 60_000

            // Set headers (pairs: [key, value, key, value, ...])
            var i = 0
            while (i + 1 < headers.size) {
                connection.setRequestProperty(headers[i], headers[i + 1])
                i += 2
            }

            // Write body
            OutputStreamWriter(connection.outputStream, Charsets.UTF_8).use { writer ->
                writer.write(body)
            }

            // Read response
            val responseCode = connection.responseCode
            val stream = if (responseCode in 200..299) {
                connection.inputStream
            } else {
                connection.errorStream ?: connection.inputStream
            }
            val responseBody = stream.bufferedReader(Charsets.UTF_8).use { it.readText() }

            if (responseCode !in 200..299) {
                throw MemmeException.Runtime("HTTP $responseCode: $responseBody")
            }
            return responseBody
        } finally {
            connection.disconnect()
        }
    }
}

/**
 * Usage example:
 *
 *   val store = MemoryStore.newWithHttpClient(
 *       dbPath = context.getDatabasePath("memory.duckdb").absolutePath,
 *       httpClient = DefaultHttpClient(),
 *       apiKey = "sk-your-api-key",
 *       embeddingModel = "text-embedding-v3",
 *       embeddingDims = 1024u,
 *       llmBaseUrl = "https://dashscope.aliyuncs.com/compatible-mode",
 *   )
 *
 *   // Add memory
 *   val result = store.add("User prefers dark mode", "user123")
 *
 *   // Search
 *   val results = store.search("theme preference", "user123", 5u)
 */
