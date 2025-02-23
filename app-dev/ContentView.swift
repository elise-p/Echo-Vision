import SwiftUI
import AVFoundation

struct ContentView: View {
    @State private var isActive = false
    @State private var isDevMode = false
    
    var body: some View {
        VStack {
            Spacer()
            
            // Status Text
            Text(isActive ? "✅ Active" : "🚫 Idle")
                .font(.title)
                .fontWeight(.bold)
                .foregroundColor(.black)
            
            Spacer()
            
            // Start/Stop Button
            Button(action: {
                isActive.toggle()
            }) {
                Text(isActive ? "Stop" : "Start")
                    .font(.title)
                    .fontWeight(.bold)
                    .frame(width: 100, height: 100)
                    .background(isActive ? Color.white : Color.gray)
                    .foregroundColor(isActive ? .black : .white)
                    .clipShape(Circle())
                    .overlay(Circle().stroke(Color.black, lineWidth: 2))
            }
            
            Spacer()
            
            // Camera Preview (Only when active)
            if isActive {
                CameraPreview()
                    .frame(height: 200)
                    .cornerRadius(10)
                    .overlay(
                        Image(systemName: "arrow.up.left.and.arrow.down.right")
                            .padding(8)
                            .background(Color.white.opacity(0.7))
                            .clipShape(Circle())
                            .offset(x: -10, y: -10),
                        alignment: .bottomTrailing
                    )
            }
            
            Spacer()
            
            // Dev Mode Toggle
            Toggle("dev mode", isOn: $isDevMode)
                .padding()
        }
        .padding()
    }
}

// Camera Preview View
struct CameraPreview: UIViewControllerRepresentable {
    func makeUIViewController(context: Context) -> UIViewController {
        let controller = UIViewController()
        let previewLayer = AVCaptureVideoPreviewLayer(session: AVCaptureSession())
        previewLayer.videoGravity = .resizeAspectFill
        previewLayer.frame = controller.view.frame
        controller.view.layer.addSublayer(previewLayer)
        return controller
    }

    func updateUIViewController(_ uiViewController: UIViewController, context: Context) {}
}

struct ContentView_Previews: PreviewProvider {
    static var previews: some View {
        ContentView()
    }
}
