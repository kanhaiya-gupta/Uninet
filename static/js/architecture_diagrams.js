window.architectureDiagrams = {
    fnn: `
        <svg width="100%" height="200" viewBox="0 0 600 200">
            <!-- Input Layer -->
            <g transform="translate(50, 50)">
                <rect x="0" y="0" width="80" height="100" fill="#e3f2fd" stroke="#2196f3" stroke-width="2"/>
                <text x="40" y="50" text-anchor="middle" fill="#1565c0">Input Layer</text>
                <circle cx="20" cy="30" r="5" fill="#2196f3"/>
                <circle cx="20" cy="50" r="5" fill="#2196f3"/>
                <circle cx="20" cy="70" r="5" fill="#2196f3"/>
            </g>
            
            <!-- Hidden Layer 1 -->
            <g transform="translate(200, 30)">
                <rect x="0" y="0" width="80" height="140" fill="#e8f5e9" stroke="#4caf50" stroke-width="2"/>
                <text x="40" y="70" text-anchor="middle" fill="#2e7d32">Hidden Layer 1</text>
                <circle cx="20" cy="30" r="5" fill="#4caf50"/>
                <circle cx="20" cy="50" r="5" fill="#4caf50"/>
                <circle cx="20" cy="70" r="5" fill="#4caf50"/>
                <circle cx="20" cy="90" r="5" fill="#4caf50"/>
            </g>
            
            <!-- Hidden Layer 2 -->
            <g transform="translate(350, 30)">
                <rect x="0" y="0" width="80" height="140" fill="#fff3e0" stroke="#ff9800" stroke-width="2"/>
                <text x="40" y="70" text-anchor="middle" fill="#e65100">Hidden Layer 2</text>
                <circle cx="20" cy="30" r="5" fill="#ff9800"/>
                <circle cx="20" cy="50" r="5" fill="#ff9800"/>
                <circle cx="20" cy="70" r="5" fill="#ff9800"/>
                <circle cx="20" cy="90" r="5" fill="#ff9800"/>
            </g>
            
            <!-- Output Layer -->
            <g transform="translate(500, 50)">
                <rect x="0" y="0" width="80" height="100" fill="#fce4ec" stroke="#e91e63" stroke-width="2"/>
                <text x="40" y="50" text-anchor="middle" fill="#c2185b">Output Layer</text>
                <circle cx="20" cy="30" r="5" fill="#e91e63"/>
                <circle cx="20" cy="50" r="5" fill="#e91e63"/>
                <circle cx="20" cy="70" r="5" fill="#e91e63"/>
            </g>
            
            <!-- Connection Lines -->
            <g stroke="#9e9e9e" stroke-width="1">
                <!-- Input to Hidden 1 -->
                <line x1="130" y1="50" x2="200" y2="50"/>
                <line x1="130" y1="70" x2="200" y2="70"/>
                <line x1="130" y1="90" x2="200" y2="90"/>
                
                <!-- Hidden 1 to Hidden 2 -->
                <line x1="280" y1="50" x2="350" y2="50"/>
                <line x1="280" y1="70" x2="350" y2="70"/>
                <line x1="280" y1="90" x2="350" y2="90"/>
                
                <!-- Hidden 2 to Output -->
                <line x1="430" y1="50" x2="500" y2="50"/>
                <line x1="430" y1="70" x2="500" y2="70"/>
                <line x1="430" y1="90" x2="500" y2="90"/>
            </g>
        </svg>
    `,
    
    cnn: `
        <svg width="100%" height="300" viewBox="0 0 600 300">
            <!-- Input Image -->
            <g transform="translate(50, 50)">
                <rect x="0" y="0" width="100" height="100" fill="#e3f2fd" stroke="#2196f3" stroke-width="2"/>
                <text x="50" y="120" text-anchor="middle" fill="#1565c0">Input Image</text>
                <!-- Grid pattern -->
                <g stroke="#2196f3" stroke-width="0.5">
                    <line x1="0" y1="25" x2="100" y2="25"/>
                    <line x1="0" y1="50" x2="100" y2="50"/>
                    <line x1="0" y1="75" x2="100" y2="75"/>
                    <line x1="25" y1="0" x2="25" y2="100"/>
                    <line x1="50" y1="0" x2="50" y2="100"/>
                    <line x1="75" y1="0" x2="75" y2="100"/>
                </g>
            </g>
            
            <!-- Convolutional Layer -->
            <g transform="translate(200, 50)">
                <rect x="0" y="0" width="100" height="100" fill="#e8f5e9" stroke="#4caf50" stroke-width="2"/>
                <text x="50" y="120" text-anchor="middle" fill="#2e7d32">Conv Layer</text>
                <!-- Feature maps -->
                <rect x="10" y="10" width="30" height="30" fill="#81c784" stroke="#4caf50"/>
                <rect x="60" y="10" width="30" height="30" fill="#81c784" stroke="#4caf50"/>
                <rect x="10" y="60" width="30" height="30" fill="#81c784" stroke="#4caf50"/>
                <rect x="60" y="60" width="30" height="30" fill="#81c784" stroke="#4caf50"/>
            </g>
            
            <!-- Pooling Layer -->
            <g transform="translate(350, 50)">
                <rect x="0" y="0" width="100" height="100" fill="#fff3e0" stroke="#ff9800" stroke-width="2"/>
                <text x="50" y="120" text-anchor="middle" fill="#e65100">Pooling</text>
                <!-- Pooled features -->
                <rect x="20" y="20" width="60" height="60" fill="#ffb74d" stroke="#ff9800"/>
            </g>
            
            <!-- Fully Connected Layer -->
            <g transform="translate(500, 50)">
                <rect x="0" y="0" width="100" height="100" fill="#fce4ec" stroke="#e91e63" stroke-width="2"/>
                <text x="50" y="120" text-anchor="middle" fill="#c2185b">Fully Connected</text>
                <circle cx="25" cy="30" r="5" fill="#e91e63"/>
                <circle cx="25" cy="50" r="5" fill="#e91e63"/>
                <circle cx="25" cy="70" r="5" fill="#e91e63"/>
                <circle cx="75" cy="30" r="5" fill="#e91e63"/>
                <circle cx="75" cy="50" r="5" fill="#e91e63"/>
                <circle cx="75" cy="70" r="5" fill="#e91e63"/>
            </g>
            
            <!-- Connection Lines -->
            <g stroke="#9e9e9e" stroke-width="1">
                <line x1="150" y1="100" x2="200" y2="100"/>
                <line x1="300" y1="100" x2="350" y2="100"/>
                <line x1="450" y1="100" x2="500" y2="100"/>
            </g>
        </svg>
    `,
    
    rnn: `
        <svg width="100%" height="200" viewBox="0 0 600 200">
            <!-- Input Sequence -->
            <g transform="translate(50, 50)">
                <rect x="0" y="0" width="80" height="100" fill="#e3f2fd" stroke="#2196f3" stroke-width="2"/>
                <text x="40" y="50" text-anchor="middle" fill="#1565c0">Input</text>
                <text x="40" y="70" text-anchor="middle" fill="#1565c0">Sequence</text>
            </g>
            
            <!-- RNN Cells -->
            <g transform="translate(200, 50)">
                <rect x="0" y="0" width="80" height="100" fill="#e8f5e9" stroke="#4caf50" stroke-width="2"/>
                <text x="40" y="50" text-anchor="middle" fill="#2e7d32">RNN</text>
                <text x="40" y="70" text-anchor="middle" fill="#2e7d32">Cell</text>
                <circle cx="40" cy="40" r="15" fill="#81c784" stroke="#4caf50"/>
            </g>
            
            <!-- Recurrent Connection -->
            <path d="M 240 100 L 240 120 L 200 120 L 200 100" 
                  fill="none" stroke="#4caf50" stroke-width="2" 
                  marker-end="url(#arrowhead)"/>
            
            <!-- Output -->
            <g transform="translate(350, 50)">
                <rect x="0" y="0" width="80" height="100" fill="#fce4ec" stroke="#e91e63" stroke-width="2"/>
                <text x="40" y="50" text-anchor="middle" fill="#c2185b">Output</text>
                <text x="40" y="70" text-anchor="middle" fill="#c2185b">Sequence</text>
            </g>
            
            <!-- Connection Lines -->
            <g stroke="#9e9e9e" stroke-width="1">
                <line x1="130" y1="100" x2="200" y2="100"/>
                <line x1="280" y1="100" x2="350" y2="100"/>
            </g>
            
            <!-- Arrow Marker -->
            <defs>
                <marker id="arrowhead" markerWidth="10" markerHeight="7" 
                        refX="9" refY="3.5" orient="auto">
                    <polygon points="0 0, 10 3.5, 0 7" fill="#4caf50"/>
                </marker>
            </defs>
        </svg>
    `,
    
    transformer: `
        <svg width="100%" height="300" viewBox="0 0 600 300">
            <!-- Input Embedding -->
            <g transform="translate(50, 50)">
                <rect x="0" y="0" width="100" height="100" fill="#e3f2fd" stroke="#2196f3" stroke-width="2"/>
                <text x="50" y="50" text-anchor="middle" fill="#1565c0">Input</text>
                <text x="50" y="70" text-anchor="middle" fill="#1565c0">Embedding</text>
            </g>
            
            <!-- Multi-Head Attention -->
            <g transform="translate(200, 50)">
                <rect x="0" y="0" width="100" height="100" fill="#e8f5e9" stroke="#4caf50" stroke-width="2"/>
                <text x="50" y="50" text-anchor="middle" fill="#2e7d32">Multi-Head</text>
                <text x="50" y="70" text-anchor="middle" fill="#2e7d32">Attention</text>
                <!-- Attention heads -->
                <circle cx="25" cy="30" r="5" fill="#81c784"/>
                <circle cx="50" cy="30" r="5" fill="#81c784"/>
                <circle cx="75" cy="30" r="5" fill="#81c784"/>
            </g>
            
            <!-- Feed Forward -->
            <g transform="translate(350, 50)">
                <rect x="0" y="0" width="100" height="100" fill="#fff3e0" stroke="#ff9800" stroke-width="2"/>
                <text x="50" y="50" text-anchor="middle" fill="#e65100">Feed</text>
                <text x="50" y="70" text-anchor="middle" fill="#e65100">Forward</text>
                <line x1="20" y1="40" x2="80" y2="40" stroke="#ff9800" stroke-width="2"/>
                <line x1="20" y1="60" x2="80" y2="60" stroke="#ff9800" stroke-width="2"/>
            </g>
            
            <!-- Output -->
            <g transform="translate(500, 50)">
                <rect x="0" y="0" width="100" height="100" fill="#fce4ec" stroke="#e91e63" stroke-width="2"/>
                <text x="50" y="50" text-anchor="middle" fill="#c2185b">Output</text>
                <text x="50" y="70" text-anchor="middle" fill="#c2185b">Layer</text>
            </g>
            
            <!-- Connection Lines -->
            <g stroke="#9e9e9e" stroke-width="1">
                <line x1="150" y1="100" x2="200" y2="100"/>
                <line x1="300" y1="100" x2="350" y2="100"/>
                <line x1="450" y1="100" x2="500" y2="100"/>
            </g>
            
            <!-- Layer Norm Indicators -->
            <g transform="translate(0, 170)">
                <text x="50" y="0" text-anchor="middle" fill="#9e9e9e">Layer Norm</text>
                <text x="200" y="0" text-anchor="middle" fill="#9e9e9e">Layer Norm</text>
                <text x="350" y="0" text-anchor="middle" fill="#9e9e9e">Layer Norm</text>
            </g>
        </svg>
    `,
    
    autoencoder: `
        <svg width="100%" height="300" viewBox="0 0 600 300">
            <!-- Encoder -->
            <g transform="translate(50, 50)">
                <rect x="0" y="0" width="100" height="200" fill="#e3f2fd" stroke="#2196f3" stroke-width="2"/>
                <text x="50" y="20" text-anchor="middle" fill="#1565c0">Encoder</text>
                <circle cx="50" cy="60" r="8" fill="#2196f3"/>
                <circle cx="50" cy="100" r="8" fill="#2196f3"/>
                <circle cx="50" cy="140" r="8" fill="#2196f3"/>
                <circle cx="50" cy="180" r="8" fill="#2196f3"/>
            </g>
            
            <!-- Latent Space -->
            <g transform="translate(250, 100)">
                <rect x="0" y="0" width="100" height="100" fill="#e8f5e9" stroke="#4caf50" stroke-width="2"/>
                <text x="50" y="50" text-anchor="middle" fill="#2e7d32">Latent Space</text>
                <circle cx="50" cy="50" r="15" fill="#81c784" stroke="#4caf50"/>
            </g>
            
            <!-- Decoder -->
            <g transform="translate(450, 50)">
                <rect x="0" y="0" width="100" height="200" fill="#fce4ec" stroke="#e91e63" stroke-width="2"/>
                <text x="50" y="20" text-anchor="middle" fill="#c2185b">Decoder</text>
                <circle cx="50" cy="60" r="8" fill="#e91e63"/>
                <circle cx="50" cy="100" r="8" fill="#e91e63"/>
                <circle cx="50" cy="140" r="8" fill="#e91e63"/>
                <circle cx="50" cy="180" r="8" fill="#e91e63"/>
            </g>
            
            <!-- Connection Lines -->
            <g stroke="#9e9e9e" stroke-width="1">
                <line x1="150" y1="100" x2="250" y2="100"/>
                <line x1="150" y1="140" x2="250" y2="140"/>
                <line x1="350" y1="100" x2="450" y2="100"/>
                <line x1="350" y1="140" x2="450" y2="140"/>
            </g>
        </svg>
    `,
    
    vae: `
        <svg width="100%" height="300" viewBox="0 0 600 300">
            <!-- Encoder -->
            <g transform="translate(50, 50)">
                <rect x="0" y="0" width="100" height="200" fill="#e3f2fd" stroke="#2196f3" stroke-width="2"/>
                <text x="50" y="20" text-anchor="middle" fill="#1565c0">Encoder</text>
                <circle cx="50" cy="60" r="8" fill="#2196f3"/>
                <circle cx="50" cy="100" r="8" fill="#2196f3"/>
                <circle cx="50" cy="140" r="8" fill="#2196f3"/>
                <circle cx="50" cy="180" r="8" fill="#2196f3"/>
            </g>
            
            <!-- Latent Space (μ and σ) -->
            <g transform="translate(250, 50)">
                <rect x="0" y="0" width="100" height="200" fill="#e8f5e9" stroke="#4caf50" stroke-width="2"/>
                <text x="50" y="20" text-anchor="middle" fill="#2e7d32">Latent Space</text>
                <text x="50" y="40" text-anchor="middle" fill="#2e7d32">(μ, σ)</text>
                <circle cx="30" cy="100" r="15" fill="#81c784" stroke="#4caf50"/>
                <text x="30" y="105" text-anchor="middle" fill="#2e7d32">μ</text>
                <circle cx="70" cy="100" r="15" fill="#81c784" stroke="#4caf50"/>
                <text x="70" y="105" text-anchor="middle" fill="#2e7d32">σ</text>
            </g>
            
            <!-- Decoder -->
            <g transform="translate(450, 50)">
                <rect x="0" y="0" width="100" height="200" fill="#fce4ec" stroke="#e91e63" stroke-width="2"/>
                <text x="50" y="20" text-anchor="middle" fill="#c2185b">Decoder</text>
                <circle cx="50" cy="60" r="8" fill="#e91e63"/>
                <circle cx="50" cy="100" r="8" fill="#e91e63"/>
                <circle cx="50" cy="140" r="8" fill="#e91e63"/>
                <circle cx="50" cy="180" r="8" fill="#e91e63"/>
            </g>
            
            <!-- Connection Lines -->
            <g stroke="#9e9e9e" stroke-width="1">
                <line x1="150" y1="100" x2="250" y2="100"/>
                <line x1="150" y1="140" x2="250" y2="140"/>
                <line x1="350" y1="100" x2="450" y2="100"/>
                <line x1="350" y1="140" x2="450" y2="140"/>
            </g>
            
            <!-- KL Divergence -->
            <text x="300" y="280" text-anchor="middle" fill="#9e9e9e">KL Divergence</text>
        </svg>
    `,
    
    gan: `
        <svg width="100%" height="300" viewBox="0 0 600 300">
            <!-- Generator -->
            <g transform="translate(50, 50)">
                <rect x="0" y="0" width="150" height="200" fill="#e3f2fd" stroke="#2196f3" stroke-width="2"/>
                <text x="75" y="20" text-anchor="middle" fill="#1565c0">Generator</text>
                <circle cx="75" cy="60" r="8" fill="#2196f3"/>
                <circle cx="75" cy="100" r="8" fill="#2196f3"/>
                <circle cx="75" cy="140" r="8" fill="#2196f3"/>
                <circle cx="75" cy="180" r="8" fill="#2196f3"/>
            </g>
            
            <!-- Generated Data -->
            <g transform="translate(250, 100)">
                <rect x="0" y="0" width="100" height="100" fill="#e8f5e9" stroke="#4caf50" stroke-width="2"/>
                <text x="50" y="50" text-anchor="middle" fill="#2e7d32">Generated</text>
                <text x="50" y="70" text-anchor="middle" fill="#2e7d32">Data</text>
            </g>
            
            <!-- Discriminator -->
            <g transform="translate(400, 50)">
                <rect x="0" y="0" width="150" height="200" fill="#fce4ec" stroke="#e91e63" stroke-width="2"/>
                <text x="75" y="20" text-anchor="middle" fill="#c2185b">Discriminator</text>
                <circle cx="75" cy="60" r="8" fill="#e91e63"/>
                <circle cx="75" cy="100" r="8" fill="#e91e63"/>
                <circle cx="75" cy="140" r="8" fill="#e91e63"/>
                <circle cx="75" cy="180" r="8" fill="#e91e63"/>
            </g>
            
            <!-- Connection Lines -->
            <g stroke="#9e9e9e" stroke-width="1">
                <line x1="200" y1="150" x2="250" y2="150"/>
                <line x1="350" y1="150" x2="400" y2="150"/>
            </g>
            
            <!-- Labels -->
            <text x="225" y="90" text-anchor="middle" fill="#9e9e9e">Generates</text>
            <text x="375" y="90" text-anchor="middle" fill="#9e9e9e">Classifies</text>
        </svg>
    `,
    
    pinn: `
        <svg width="100%" height="300" viewBox="0 0 600 300">
            <!-- Neural Network -->
            <g transform="translate(50, 50)">
                <rect x="0" y="0" width="150" height="200" fill="#e3f2fd" stroke="#2196f3" stroke-width="2"/>
                <text x="75" y="20" text-anchor="middle" fill="#1565c0">Neural Network</text>
                <circle cx="75" cy="60" r="8" fill="#2196f3"/>
                <circle cx="75" cy="100" r="8" fill="#2196f3"/>
                <circle cx="75" cy="140" r="8" fill="#2196f3"/>
                <circle cx="75" cy="180" r="8" fill="#2196f3"/>
            </g>
            
            <!-- Physics Constraints -->
            <g transform="translate(250, 50)">
                <rect x="0" y="0" width="100" height="200" fill="#e8f5e9" stroke="#4caf50" stroke-width="2"/>
                <text x="50" y="20" text-anchor="middle" fill="#2e7d32">Physics</text>
                <text x="50" y="40" text-anchor="middle" fill="#2e7d32">Constraints</text>
                <path d="M 20 80 L 80 80 M 20 120 L 80 120" 
                      stroke="#4caf50" stroke-width="2"/>
                <text x="50" y="100" text-anchor="middle" fill="#2e7d32">PDEs</text>
                <text x="50" y="140" text-anchor="middle" fill="#2e7d32">BCs</text>
                <text x="50" y="160" text-anchor="middle" fill="#2e7d32">ICs</text>
            </g>
            
            <!-- Loss Function -->
            <g transform="translate(400, 50)">
                <rect x="0" y="0" width="150" height="200" fill="#fce4ec" stroke="#e91e63" stroke-width="2"/>
                <text x="75" y="20" text-anchor="middle" fill="#c2185b">Loss Function</text>
                <text x="75" y="60" text-anchor="middle" fill="#c2185b">L = L_data +</text>
                <text x="75" y="80" text-anchor="middle" fill="#c2185b">λ * L_physics</text>
                <circle cx="75" cy="140" r="30" fill="#f8bbd0" stroke="#e91e63"/>
                <text x="75" y="145" text-anchor="middle" fill="#c2185b">L</text>
            </g>
            
            <!-- Connection Lines -->
            <g stroke="#9e9e9e" stroke-width="1">
                <line x1="200" y1="150" x2="250" y2="150"/>
                <line x1="350" y1="150" x2="400" y2="150"/>
            </g>
            
            <!-- Labels -->
            <text x="225" y="90" text-anchor="middle" fill="#9e9e9e">Satisfies</text>
            <text x="375" y="90" text-anchor="middle" fill="#9e9e9e">Minimizes</text>
        </svg>
    `,
    
    gnn: `
        <svg width="100%" height="300" viewBox="0 0 600 300">
            <!-- Graph Structure -->
            <g transform="translate(50, 50)">
                <rect x="0" y="0" width="150" height="200" fill="#e3f2fd" stroke="#2196f3" stroke-width="2"/>
                <text x="75" y="20" text-anchor="middle" fill="#1565c0">Graph Structure</text>
                <!-- Nodes -->
                <circle cx="40" cy="60" r="10" fill="#2196f3"/>
                <circle cx="110" cy="60" r="10" fill="#2196f3"/>
                <circle cx="75" cy="120" r="10" fill="#2196f3"/>
                <circle cx="40" cy="180" r="10" fill="#2196f3"/>
                <circle cx="110" cy="180" r="10" fill="#2196f3"/>
                <!-- Edges -->
                <line x1="40" y1="60" x2="110" y2="60" stroke="#2196f3" stroke-width="2"/>
                <line x1="40" y1="60" x2="75" y2="120" stroke="#2196f3" stroke-width="2"/>
                <line x1="110" y1="60" x2="75" y2="120" stroke="#2196f3" stroke-width="2"/>
                <line x1="75" y1="120" x2="40" y2="180" stroke="#2196f3" stroke-width="2"/>
                <line x1="75" y1="120" x2="110" y2="180" stroke="#2196f3" stroke-width="2"/>
            </g>
            
            <!-- Message Passing -->
            <g transform="translate(250, 50)">
                <rect x="0" y="0" width="100" height="200" fill="#e8f5e9" stroke="#4caf50" stroke-width="2"/>
                <text x="50" y="20" text-anchor="middle" fill="#2e7d32">Message</text>
                <text x="50" y="40" text-anchor="middle" fill="#2e7d32">Passing</text>
                <!-- Arrows -->
                <path d="M 20 100 L 80 100" stroke="#4caf50" stroke-width="2" marker-end="url(#arrowhead)"/>
                <path d="M 50 80 L 50 120" stroke="#4caf50" stroke-width="2" marker-end="url(#arrowhead)"/>
            </g>
            
            <!-- Node Updates -->
            <g transform="translate(400, 50)">
                <rect x="0" y="0" width="150" height="200" fill="#fce4ec" stroke="#e91e63" stroke-width="2"/>
                <text x="75" y="20" text-anchor="middle" fill="#c2185b">Node Updates</text>
                <!-- Updated Nodes -->
                <circle cx="40" cy="60" r="10" fill="#e91e63"/>
                <circle cx="110" cy="60" r="10" fill="#e91e63"/>
                <circle cx="75" cy="120" r="10" fill="#e91e63"/>
                <circle cx="40" cy="180" r="10" fill="#e91e63"/>
                <circle cx="110" cy="180" r="10" fill="#e91e63"/>
                <!-- Updated Edges -->
                <line x1="40" y1="60" x2="110" y2="60" stroke="#e91e63" stroke-width="2"/>
                <line x1="40" y1="60" x2="75" y2="120" stroke="#e91e63" stroke-width="2"/>
                <line x1="110" y1="60" x2="75" y2="120" stroke="#e91e63" stroke-width="2"/>
                <line x1="75" y1="120" x2="40" y2="180" stroke="#e91e63" stroke-width="2"/>
                <line x1="75" y1="120" x2="110" y2="180" stroke="#e91e63" stroke-width="2"/>
            </g>
            
            <!-- Arrow Marker -->
            <defs>
                <marker id="arrowhead" markerWidth="10" markerHeight="7" 
                        refX="9" refY="3.5" orient="auto">
                    <polygon points="0 0, 10 3.5, 0 7" fill="#4caf50"/>
                </marker>
            </defs>
        </svg>
    `,
    
    snn: `
        <svg width="100%" height="300" viewBox="0 0 600 300">
            <!-- Input Layer -->
            <g transform="translate(50, 50)">
                <rect x="0" y="0" width="100" height="200" fill="#e3f2fd" stroke="#2196f3" stroke-width="2"/>
                <text x="50" y="20" text-anchor="middle" fill="#1565c0">Input</text>
                <text x="50" y="40" text-anchor="middle" fill="#1565c0">Neurons</text>
                <!-- Spikes -->
                <path d="M 30 80 L 30 120" stroke="#2196f3" stroke-width="2"/>
                <path d="M 50 80 L 50 120" stroke="#2196f3" stroke-width="2"/>
                <path d="M 70 80 L 70 120" stroke="#2196f3" stroke-width="2"/>
                <!-- Spike markers -->
                <circle cx="30" cy="80" r="3" fill="#2196f3"/>
                <circle cx="50" cy="100" r="3" fill="#2196f3"/>
                <circle cx="70" cy="120" r="3" fill="#2196f3"/>
            </g>
            
            <!-- Synaptic Connections -->
            <g transform="translate(200, 50)">
                <rect x="0" y="0" width="100" height="200" fill="#e8f5e9" stroke="#4caf50" stroke-width="2"/>
                <text x="50" y="20" text-anchor="middle" fill="#2e7d32">Synaptic</text>
                <text x="50" y="40" text-anchor="middle" fill="#2e7d32">Connections</text>
                <!-- Connection lines -->
                <path d="M 20 80 C 50 60, 50 140, 80 120" 
                      stroke="#4caf50" stroke-width="2" fill="none"/>
                <path d="M 20 120 C 50 100, 50 180, 80 160" 
                      stroke="#4caf50" stroke-width="2" fill="none"/>
            </g>
            
            <!-- Output Layer -->
            <g transform="translate(350, 50)">
                <rect x="0" y="0" width="100" height="200" fill="#fce4ec" stroke="#e91e63" stroke-width="2"/>
                <text x="50" y="20" text-anchor="middle" fill="#c2185b">Output</text>
                <text x="50" y="40" text-anchor="middle" fill="#c2185b">Neurons</text>
                <!-- Membrane potential -->
                <path d="M 30 80 L 30 120" stroke="#e91e63" stroke-width="2"/>
                <path d="M 50 80 L 50 120" stroke="#e91e63" stroke-width="2"/>
                <path d="M 70 80 L 70 120" stroke="#e91e63" stroke-width="2"/>
                <!-- Threshold line -->
                <line x1="20" y1="100" x2="80" y2="100" 
                      stroke="#e91e63" stroke-width="1" stroke-dasharray="5,5"/>
            </g>
            
            <!-- Time Steps -->
            <g transform="translate(0, 270)">
                <text x="300" y="0" text-anchor="middle" fill="#9e9e9e">Time Steps</text>
                <line x1="50" y1="0" x2="550" y2="0" stroke="#9e9e9e" stroke-width="1"/>
                <circle cx="100" cy="0" r="2" fill="#9e9e9e"/>
                <circle cx="200" cy="0" r="2" fill="#9e9e9e"/>
                <circle cx="300" cy="0" r="2" fill="#9e9e9e"/>
                <circle cx="400" cy="0" r="2" fill="#9e9e9e"/>
                <circle cx="500" cy="0" r="2" fill="#9e9e9e"/>
            </g>
        </svg>
    `,
    
    neural_odes: `
        <svg width="100%" height="300" viewBox="0 0 600 300">
            <!-- Initial State -->
            <g transform="translate(50, 50)">
                <rect x="0" y="0" width="100" height="200" fill="#e3f2fd" stroke="#2196f3" stroke-width="2"/>
                <text x="50" y="20" text-anchor="middle" fill="#1565c0">Initial</text>
                <text x="50" y="40" text-anchor="middle" fill="#1565c0">State</text>
                <circle cx="50" cy="120" r="30" fill="#2196f3" fill-opacity="0.3"/>
                <text x="50" y="125" text-anchor="middle" fill="#1565c0">x₀</text>
            </g>
            
            <!-- Neural ODE -->
            <g transform="translate(200, 50)">
                <rect x="0" y="0" width="200" height="200" fill="#e8f5e9" stroke="#4caf50" stroke-width="2"/>
                <text x="100" y="20" text-anchor="middle" fill="#2e7d32">Neural ODE</text>
                <text x="100" y="40" text-anchor="middle" fill="#2e7d32">dx/dt = f(x, t)</text>
                <!-- Flow field -->
                <path d="M 50 100 C 100 50, 150 150, 200 100" 
                      stroke="#4caf50" stroke-width="2" fill="none"/>
                <path d="M 50 150 C 100 100, 150 200, 200 150" 
                      stroke="#4caf50" stroke-width="2" fill="none"/>
                <!-- Time steps -->
                <circle cx="50" cy="100" r="5" fill="#4caf50"/>
                <circle cx="100" cy="75" r="5" fill="#4caf50"/>
                <circle cx="150" cy="125" r="5" fill="#4caf50"/>
                <circle cx="200" cy="100" r="5" fill="#4caf50"/>
            </g>
            
            <!-- Final State -->
            <g transform="translate(450, 50)">
                <rect x="0" y="0" width="100" height="200" fill="#fce4ec" stroke="#e91e63" stroke-width="2"/>
                <text x="50" y="20" text-anchor="middle" fill="#c2185b">Final</text>
                <text x="50" y="40" text-anchor="middle" fill="#c2185b">State</text>
                <circle cx="50" cy="120" r="30" fill="#e91e63" fill-opacity="0.3"/>
                <text x="50" y="125" text-anchor="middle" fill="#c2185b">x₁</text>
            </g>
            
            <!-- Time Axis -->
            <g transform="translate(0, 270)">
                <text x="300" y="0" text-anchor="middle" fill="#9e9e9e">Time (t)</text>
                <line x1="50" y1="0" x2="550" y2="0" stroke="#9e9e9e" stroke-width="1"/>
                <text x="50" y="20" text-anchor="middle" fill="#9e9e9e">t₀</text>
                <text x="550" y="20" text-anchor="middle" fill="#9e9e9e">t₁</text>
            </g>
        </svg>
    `,
    
    dbn: `
        <svg width="100%" height="300" viewBox="0 0 600 300">
            <!-- Visible Layer -->
            <g transform="translate(50, 50)">
                <rect x="0" y="0" width="100" height="200" fill="#e3f2fd" stroke="#2196f3" stroke-width="2"/>
                <text x="50" y="20" text-anchor="middle" fill="#1565c0">Visible</text>
                <text x="50" y="40" text-anchor="middle" fill="#1565c0">Layer</text>
                <circle cx="50" cy="80" r="8" fill="#2196f3"/>
                <circle cx="50" cy="120" r="8" fill="#2196f3"/>
                <circle cx="50" cy="160" r="8" fill="#2196f3"/>
            </g>
            
            <!-- Hidden Layer 1 -->
            <g transform="translate(200, 50)">
                <rect x="0" y="0" width="100" height="200" fill="#e8f5e9" stroke="#4caf50" stroke-width="2"/>
                <text x="50" y="20" text-anchor="middle" fill="#2e7d32">Hidden</text>
                <text x="50" y="40" text-anchor="middle" fill="#2e7d32">Layer 1</text>
                <circle cx="50" cy="80" r="8" fill="#4caf50"/>
                <circle cx="50" cy="120" r="8" fill="#4caf50"/>
                <circle cx="50" cy="160" r="8" fill="#4caf50"/>
            </g>
            
            <!-- Hidden Layer 2 -->
            <g transform="translate(350, 50)">
                <rect x="0" y="0" width="100" height="200" fill="#fff3e0" stroke="#ff9800" stroke-width="2"/>
                <text x="50" y="20" text-anchor="middle" fill="#e65100">Hidden</text>
                <text x="50" y="40" text-anchor="middle" fill="#e65100">Layer 2</text>
                <circle cx="50" cy="80" r="8" fill="#ff9800"/>
                <circle cx="50" cy="120" r="8" fill="#ff9800"/>
                <circle cx="50" cy="160" r="8" fill="#ff9800"/>
            </g>
            
            <!-- Connection Lines -->
            <g stroke="#9e9e9e" stroke-width="1">
                <!-- Visible to Hidden 1 -->
                <line x1="150" y1="80" x2="200" y2="80"/>
                <line x1="150" y1="120" x2="200" y2="120"/>
                <line x1="150" y1="160" x2="200" y2="160"/>
                
                <!-- Hidden 1 to Hidden 2 -->
                <line x1="300" y1="80" x2="350" y2="80"/>
                <line x1="300" y1="120" x2="350" y2="120"/>
                <line x1="300" y1="160" x2="350" y2="160"/>
            </g>
            
            <!-- Training Process -->
            <g transform="translate(0, 270)">
                <text x="300" y="0" text-anchor="middle" fill="#9e9e9e">Layer-wise Pre-training</text>
                <text x="300" y="20" text-anchor="middle" fill="#9e9e9e">Followed by Fine-tuning</text>
            </g>
        </svg>
    `,
    
    rbfn: `
        <svg width="100%" height="300" viewBox="0 0 600 300">
            <!-- Input Layer -->
            <g transform="translate(50, 50)">
                <rect x="0" y="0" width="100" height="200" fill="#e3f2fd" stroke="#2196f3" stroke-width="2"/>
                <text x="50" y="20" text-anchor="middle" fill="#1565c0">Input</text>
                <text x="50" y="40" text-anchor="middle" fill="#1565c0">Layer</text>
                <circle cx="50" cy="80" r="8" fill="#2196f3"/>
                <circle cx="50" cy="120" r="8" fill="#2196f3"/>
                <circle cx="50" cy="160" r="8" fill="#2196f3"/>
            </g>
            
            <!-- RBF Layer -->
            <g transform="translate(200, 50)">
                <rect x="0" y="0" width="200" height="200" fill="#e8f5e9" stroke="#4caf50" stroke-width="2"/>
                <text x="100" y="20" text-anchor="middle" fill="#2e7d32">RBF Layer</text>
                <!-- RBF Centers -->
                <circle cx="50" cy="100" r="20" fill="#81c784" fill-opacity="0.3"/>
                <circle cx="100" cy="100" r="20" fill="#81c784" fill-opacity="0.3"/>
                <circle cx="150" cy="100" r="20" fill="#81c784" fill-opacity="0.3"/>
                <!-- Center points -->
                <circle cx="50" cy="100" r="5" fill="#4caf50"/>
                <circle cx="100" cy="100" r="5" fill="#4caf50"/>
                <circle cx="150" cy="100" r="5" fill="#4caf50"/>
            </g>
            
            <!-- Output Layer -->
            <g transform="translate(450, 50)">
                <rect x="0" y="0" width="100" height="200" fill="#fce4ec" stroke="#e91e63" stroke-width="2"/>
                <text x="50" y="20" text-anchor="middle" fill="#c2185b">Output</text>
                <text x="50" y="40" text-anchor="middle" fill="#c2185b">Layer</text>
                <circle cx="50" cy="100" r="8" fill="#e91e63"/>
            </g>
            
            <!-- Connection Lines -->
            <g stroke="#9e9e9e" stroke-width="1">
                <line x1="150" y1="80" x2="200" y2="80"/>
                <line x1="150" y1="120" x2="200" y2="120"/>
                <line x1="150" y1="160" x2="200" y2="160"/>
                <line x1="400" y1="100" x2="450" y2="100"/>
            </g>
            
            <!-- Radial Basis Functions -->
            <g transform="translate(0, 270)">
                <text x="300" y="0" text-anchor="middle" fill="#9e9e9e">Radial Basis Functions</text>
                <text x="300" y="20" text-anchor="middle" fill="#9e9e9e">φ(x) = exp(-||x-c||²/2σ²)</text>
            </g>
        </svg>
    `
}; 