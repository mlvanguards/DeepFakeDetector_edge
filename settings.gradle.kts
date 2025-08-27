pluginManagement {
    repositories {
        google {
            content {
                includeGroupByRegex("com\\.android.*")
                includeGroupByRegex("com\\.google.*")
                includeGroupByRegex("androidx.*")
            }
        }
        mavenCentral()
        gradlePluginPortal()
    }
    plugins {
        // make sure these versions match whatever AGP / Kotlin you're on
        id("com.android.application")      version "8.1.0"
        id("com.android.asset-pack")       version "8.1.0"
        kotlin("android")                  version "1.8.10"
    }
}

dependencyResolutionManagement {
    repositoriesMode.set(RepositoriesMode.FAIL_ON_PROJECT_REPOS)
    repositories {
        google()
        mavenCentral()
    }
}

rootProject.name = "BYOS"
// Single module app (remove missing asset-pack module)
include(":app")
