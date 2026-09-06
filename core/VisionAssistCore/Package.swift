// swift-tools-version: 5.9
import PackageDescription

let package = Package(
    name: "VisionAssistCore",
    platforms: [.iOS(.v17), .macOS(.v14)],
    products: [
        .library(name: "Masking", targets: ["Masking"]),
    ],
    targets: [
        .target(name: "Masking"),
        .testTarget(name: "MaskingTests", dependencies: ["Masking"]),
    ]
)
