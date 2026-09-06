// swift-tools-version: 5.9
import PackageDescription

let package = Package(
    name: "VisionAssistCore",
    platforms: [.iOS(.v17), .macOS(.v14)],
    products: [
        .library(name: "Masking", targets: ["Masking"]),
        .library(name: "YOLODecoding", targets: ["YOLODecoding"]),
    ],
    targets: [
        .target(name: "Masking"),
        .target(name: "YOLODecoding", dependencies: ["Masking"]),
        .testTarget(name: "MaskingTests", dependencies: ["Masking"]),
        .testTarget(name: "YOLODecodingTests", dependencies: ["YOLODecoding"]),
    ]
)
