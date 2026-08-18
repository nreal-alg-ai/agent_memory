import java.net.URI
import java.nio.file.Files
import java.nio.file.StandardCopyOption
import java.security.MessageDigest

plugins {
    id("com.android.application")
    id("org.jetbrains.kotlin.android")
    id("com.chaquo.python")
}

val repositoryRoot = rootProject.projectDir.parentFile
val sherpaVersion = "1.13.4"
val sherpaSha256 = "03f9c4df965f21c71269365a7951a7f23b5696fddd093fa318c80d65550ab780"
val sherpaAar = layout.buildDirectory.file("verified-dependencies/sherpa-onnx-$sherpaVersion.aar")
val fetchSherpaAar = tasks.register("fetchSherpaAar") {
    outputs.file(sherpaAar)
    doLast {
        val target = sherpaAar.get().asFile
        fun sha256(file: File): String {
            val digest = MessageDigest.getInstance("SHA-256")
            file.inputStream().buffered().use { input ->
                val buffer = ByteArray(DEFAULT_BUFFER_SIZE)
                while (true) {
                    val count = input.read(buffer)
                    if (count < 0) break
                    digest.update(buffer, 0, count)
                }
            }
            return digest.digest().joinToString("") { "%02x".format(it) }
        }
        if (target.isFile && sha256(target) == sherpaSha256) return@doLast
        target.parentFile.mkdirs()
        val temporary = File(target.parentFile, ".${target.name}.part")
        temporary.delete()
        URI("https://github.com/k2-fsa/sherpa-onnx/releases/download/v$sherpaVersion/sherpa-onnx-$sherpaVersion.aar")
            .toURL()
            .openStream()
            .buffered()
            .use { input -> temporary.outputStream().buffered().use(input::copyTo) }
        val actualSha256 = sha256(temporary)
        check(actualSha256 == sherpaSha256) {
            temporary.delete()
            "sherpa-onnx AAR checksum mismatch: $actualSha256"
        }
        Files.move(
            temporary.toPath(),
            target.toPath(),
            StandardCopyOption.ATOMIC_MOVE,
            StandardCopyOption.REPLACE_EXISTING,
        )
    }
}

android {
    namespace = "com.agentmemory.test"
    compileSdk = 35

    defaultConfig {
        applicationId = "com.agentmemory.test"
        minSdk = 26
        targetSdk = 35
        versionCode = 1
        versionName = "0.1.0"
        testInstrumentationRunner = "androidx.test.runner.AndroidJUnitRunner"
        ndk {
            abiFilters += listOf("arm64-v8a")
        }
    }

    buildTypes {
        release {
            isMinifyEnabled = false
        }
    }

    sourceSets.getByName("main").assets.srcDir(repositoryRoot.resolve("static"))

    buildFeatures {
        buildConfig = true
    }

    compileOptions {
        sourceCompatibility = JavaVersion.VERSION_17
        targetCompatibility = JavaVersion.VERSION_17
    }
    kotlinOptions {
        jvmTarget = "17"
    }
}

chaquopy {
    defaultConfig {
        version = "3.11"
        pip {
            install("numpy==1.26.2")
            install("pyyaml")
            install("requests")
        }
    }
    sourceSets {
        getByName("main") {
            // 项目根作为模块根：backend/ -> backend.* 包；仓库 src/ 内容根：memory.* 包
            srcDir(repositoryRoot)
            srcDir(repositoryRoot.resolve("../src"))
            include("backend/**/*.py")
            include("memory/**/*.py")
        }
    }
}

fun registerChaquopyFts5Task(variant: String) {
    val variantName = variant.replaceFirstChar { it.uppercase() }
    val task = tasks.register("enableChaquopyFts5$variantName") {
        dependsOn("generate${variantName}PythonJniLibs", "generate${variantName}PythonMiscAssets")
        outputs.upToDateWhen { false }
        doLast {
            val targetCache = File(System.getProperty("user.home"), ".gradle/caches/modules-2/files-2.1/com.chaquo.python/target")
            val targetZip = fileTree(targetCache).matching {
                include("**/target-3.11.*-arm64-v8a.zip")
            }.files.maxByOrNull { it.lastModified() }
                ?: error("Chaquopy Python target archive was not found under $targetCache")
            val ndkRoot = sequenceOf(
                System.getenv("ANDROID_NDK_HOME")?.let(::File),
                System.getenv("ANDROID_HOME")?.let { File(it, "ndk") },
                File(System.getProperty("user.home"), "Library/Android/sdk/ndk"),
                File("/opt/homebrew/share/android-commandlinetools/ndk"),
            ).filterNotNull()
                .flatMap { root ->
                    if (root.resolve("toolchains/llvm").isDirectory) sequenceOf(root)
                    else root.listFiles()?.asSequence().orEmpty()
                }
                .filter { it.resolve("toolchains/llvm").isDirectory }
                .maxByOrNull { it.name }
                ?: error("Android NDK was not found")
            exec {
                commandLine(
                    "python3",
                    File(rootProject.projectDir, "tools/build_fts5_sqlite.py"),
                    "--cache-dir", layout.buildDirectory.dir("fts5-runtime").get().asFile,
                    "--target-zip", targetZip,
                    "--stdlib-imy", layout.buildDirectory.file("python/assets/misc/$variant/chaquopy/stdlib-arm64-v8a.imy").get().asFile,
                    "--output", layout.buildDirectory.file("fts5-runtime/custom/_sqlite3.cpython-311.so").get().asFile,
                    "--ndk-root", ndkRoot,
                    "--abi", "arm64-v8a",
                    "--api", "26",
                )
            }
        }
    }
    tasks.matching {
        it.name == "merge${variantName}Assets" || it.name == "package${variantName}"
    }.configureEach {
        dependsOn(task)
        outputs.upToDateWhen { false }
    }
}

registerChaquopyFts5Task("debug")
registerChaquopyFts5Task("release")

dependencies {
    implementation(files(fetchSherpaAar))
    testImplementation("junit:junit:4.13.2")
    testImplementation("org.json:json:20240303")
    androidTestImplementation("androidx.test.ext:junit:1.3.0")
    androidTestImplementation("androidx.test:runner:1.7.0")
}
