window.implementationExamples = {
    fnn: {
        classification: `
import torch
import torch.nn as nn
import torch.optim as optim

class FNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(FNN, self).__init__()
        self.layer1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.layer2 = nn.Linear(hidden_size, num_classes)
    
    def forward(self, x):
        x = self.layer1(x)
        x = self.relu(x)
        x = self.layer2(x)
        return x

# Training setup
model = FNN(input_size=784, hidden_size=128, num_classes=10)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop
for epoch in range(num_epochs):
    for inputs, labels in train_loader:
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
`,
        regression: `
import torch
import torch.nn as nn
import torch.optim as optim

class FNN(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(FNN, self).__init__()
        self.layer1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.layer2 = nn.Linear(hidden_size, 1)
    
    def forward(self, x):
        x = self.layer1(x)
        x = self.relu(x)
        x = self.layer2(x)
        return x

# Training setup
model = FNN(input_size=10, hidden_size=64)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop
for epoch in range(num_epochs):
    for inputs, targets in train_loader:
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
`
    },
    
    cnn: {
        classification: `
import torch
import torch.nn as nn
import torch.optim as optim

class CNN(nn.Module):
    def __init__(self, num_classes):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3)
        self.fc1 = nn.Linear(64 * 5 * 5, 128)
        self.fc2 = nn.Linear(128, num_classes)
    
    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = x.view(-1, 64 * 5 * 5)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# Training setup
model = CNN(num_classes=10)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop
for epoch in range(num_epochs):
    for images, labels in train_loader:
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
`,
        object_detection: `
import torch
import torch.nn as nn
import torch.optim as optim

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.backbone = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )
        self.regression_head = nn.Sequential(
            nn.Linear(128 * 56 * 56, 512),
            nn.ReLU(),
            nn.Linear(512, 4)  # x, y, w, h
        )
        self.classification_head = nn.Sequential(
            nn.Linear(128 * 56 * 56, 512),
            nn.ReLU(),
            nn.Linear(512, num_classes)
        )
    
    def forward(self, x):
        features = self.backbone(x)
        features = features.view(features.size(0), -1)
        boxes = self.regression_head(features)
        classes = self.classification_head(features)
        return boxes, classes

# Training setup
model = CNN()
criterion = nn.MSELoss()  # For boxes
class_criterion = nn.CrossEntropyLoss()  # For classes
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop
for epoch in range(num_epochs):
    for images, boxes, classes in train_loader:
        optimizer.zero_grad()
        pred_boxes, pred_classes = model(images)
        box_loss = criterion(pred_boxes, boxes)
        class_loss = class_criterion(pred_classes, classes)
        loss = box_loss + class_loss
        loss.backward()
        optimizer.step()
`
    },
    
    rnn: {
        sequence_modeling: `
import torch
import torch.nn as nn
import torch.optim as optim

class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers):
        super(RNN, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, input_size)
    
    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out)
        return out

# Training setup
model = RNN(input_size=64, hidden_size=128, num_layers=2)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop
for epoch in range(num_epochs):
    for sequences in train_loader:
        optimizer.zero_grad()
        outputs = model(sequences)
        loss = criterion(outputs, sequences)
        loss.backward()
        optimizer.step()
`,
        time_series: `
import torch
import torch.nn as nn
import torch.optim as optim

class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(RNN, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.gru = nn.GRU(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
    
    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        out, _ = self.gru(x, h0)
        out = self.fc(out[:, -1, :])
        return out

# Training setup
model = RNN(input_size=1, hidden_size=64, num_layers=2, output_size=1)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop
for epoch in range(num_epochs):
    for inputs, targets in train_loader:
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
`
    },
    
    transformer: {
        nlp: `
import torch
import torch.nn as nn
import torch.optim as optim

class Transformer(nn.Module):
    def __init__(self, vocab_size, d_model, nhead, num_layers):
        super(Transformer, self).__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoder = PositionalEncoding(d_model)
        encoder_layers = nn.TransformerEncoderLayer(d_model, nhead)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers)
        self.decoder = nn.Linear(d_model, vocab_size)
    
    def forward(self, src):
        src = self.embedding(src)
        src = self.pos_encoder(src)
        output = self.transformer_encoder(src)
        output = self.decoder(output)
        return output

# Training setup
model = Transformer(vocab_size=10000, d_model=512, nhead=8, num_layers=6)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop
for epoch in range(num_epochs):
    for inputs, targets in train_loader:
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs.view(-1, vocab_size), targets.view(-1))
        loss.backward()
        optimizer.step()
`,
        translation: `
import torch
import torch.nn as nn
import torch.optim as optim

class Transformer(nn.Module):
    def __init__(self, src_vocab_size, tgt_vocab_size, d_model, nhead, num_layers):
        super(Transformer, self).__init__()
        self.encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model, nhead),
            num_layers
        )
        self.decoder = nn.TransformerDecoder(
            nn.TransformerDecoderLayer(d_model, nhead),
            num_layers
        )
        self.src_embedding = nn.Embedding(src_vocab_size, d_model)
        self.tgt_embedding = nn.Embedding(tgt_vocab_size, d_model)
        self.pos_encoder = PositionalEncoding(d_model)
        self.fc = nn.Linear(d_model, tgt_vocab_size)
    
    def forward(self, src, tgt):
        src = self.pos_encoder(self.src_embedding(src))
        tgt = self.pos_encoder(self.tgt_embedding(tgt))
        memory = self.encoder(src)
        output = self.decoder(tgt, memory)
        return self.fc(output)

# Training setup
model = Transformer(src_vocab_size=10000, tgt_vocab_size=10000, 
                   d_model=512, nhead=8, num_layers=6)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop
for epoch in range(num_epochs):
    for src, tgt in train_loader:
        optimizer.zero_grad()
        outputs = model(src, tgt[:, :-1])
        loss = criterion(outputs.view(-1, tgt_vocab_size), tgt[:, 1:].view(-1))
        loss.backward()
        optimizer.step()
`
    },
    
    autoencoder: {
        compression: `
import torch
import torch.nn as nn
import torch.optim as optim

class Autoencoder(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(Autoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_size, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, hidden_size)
        )
        self.decoder = nn.Sequential(
            nn.Linear(hidden_size, 256),
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, input_size),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

# Training setup
model = Autoencoder(input_size=784, hidden_size=32)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop
for epoch in range(num_epochs):
    for data in train_loader:
        optimizer.zero_grad()
        outputs = model(data)
        loss = criterion(outputs, data)
        loss.backward()
        optimizer.step()
`,
        denoising: `
import torch
import torch.nn as nn
import torch.optim as optim

class DenoisingAutoencoder(nn.Module):
    def __init__(self, input_size):
        super(DenoisingAutoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_size, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 128)
        )
        self.decoder = nn.Sequential(
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, input_size),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

# Training setup
model = DenoisingAutoencoder(input_size=784)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop
for epoch in range(num_epochs):
    for clean_data in train_loader:
        # Add noise to input
        noisy_data = clean_data + torch.randn_like(clean_data) * 0.1
        optimizer.zero_grad()
        outputs = model(noisy_data)
        loss = criterion(outputs, clean_data)
        loss.backward()
        optimizer.step()
`
    },
    
    vae: {
        generation: `
import torch
import torch.nn as nn
import torch.optim as optim

class VAE(nn.Module):
    def __init__(self, input_size, hidden_size, latent_size):
        super(VAE, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU()
        )
        self.fc_mu = nn.Linear(hidden_size, latent_size)
        self.fc_var = nn.Linear(hidden_size, latent_size)
        self.decoder = nn.Sequential(
            nn.Linear(latent_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, input_size),
            nn.Sigmoid()
        )
    
    def encode(self, x):
        h = self.encoder(x)
        return self.fc_mu(h), self.fc_var(h)
    
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def decode(self, z):
        return self.decoder(z)
    
    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar

# Training setup
model = VAE(input_size=784, hidden_size=400, latent_size=20)
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop
for epoch in range(num_epochs):
    for data in train_loader:
        optimizer.zero_grad()
        recon_batch, mu, logvar = model(data)
        recon_loss = nn.MSELoss()(recon_batch, data)
        kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        loss = recon_loss + kl_loss
        loss.backward()
        optimizer.step()
`,
        style_transfer: `
import torch
import torch.nn as nn
import torch.optim as optim

class StyleVAE(nn.Module):
    def __init__(self, input_size, style_size, content_size):
        super(StyleVAE, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_size, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU()
        )
        self.fc_style_mu = nn.Linear(256, style_size)
        self.fc_style_var = nn.Linear(256, style_size)
        self.fc_content = nn.Linear(256, content_size)
        self.decoder = nn.Sequential(
            nn.Linear(style_size + content_size, 256),
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, input_size),
            nn.Sigmoid()
        )
    
    def encode(self, x):
        h = self.encoder(x)
        style_mu = self.fc_style_mu(h)
        style_var = self.fc_style_var(h)
        content = self.fc_content(h)
        return style_mu, style_var, content
    
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def decode(self, style, content):
        return self.decoder(torch.cat([style, content], dim=1))
    
    def forward(self, x):
        style_mu, style_var, content = self.encode(x)
        style = self.reparameterize(style_mu, style_var)
        return self.decode(style, content), style_mu, style_var

# Training setup
model = StyleVAE(input_size=784, style_size=10, content_size=10)
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop
for epoch in range(num_epochs):
    for data in train_loader:
        optimizer.zero_grad()
        recon_batch, style_mu, style_var = model(data)
        recon_loss = nn.MSELoss()(recon_batch, data)
        kl_loss = -0.5 * torch.sum(1 + style_var - style_mu.pow(2) - style_var.exp())
        loss = recon_loss + kl_loss
        loss.backward()
        optimizer.step()
`
    },
    
    gan: {
        generation: `
import torch
import torch.nn as nn
import torch.optim as optim

class Generator(nn.Module):
    def __init__(self, latent_dim, output_size):
        super(Generator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(latent_dim, 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 512),
            nn.LeakyReLU(0.2),
            nn.Linear(512, output_size),
            nn.Tanh()
        )
    
    def forward(self, z):
        return self.model(z)

class Discriminator(nn.Module):
    def __init__(self, input_size):
        super(Discriminator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_size, 512),
            nn.LeakyReLU(0.2),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        return self.model(x)

# Training setup
generator = Generator(latent_dim=100, output_size=784)
discriminator = Discriminator(input_size=784)
g_optimizer = optim.Adam(generator.parameters(), lr=0.0002, betas=(0.5, 0.999))
d_optimizer = optim.Adam(discriminator.parameters(), lr=0.0002, betas=(0.5, 0.999))
criterion = nn.BCELoss()

# Training loop
for epoch in range(num_epochs):
    for real_data in train_loader:
        batch_size = real_data.size(0)
        
        # Train Discriminator
        d_optimizer.zero_grad()
        label_real = torch.ones(batch_size, 1)
        label_fake = torch.zeros(batch_size, 1)
        
        output_real = discriminator(real_data)
        d_loss_real = criterion(output_real, label_real)
        
        noise = torch.randn(batch_size, 100)
        fake_data = generator(noise)
        output_fake = discriminator(fake_data.detach())
        d_loss_fake = criterion(output_fake, label_fake)
        
        d_loss = d_loss_real + d_loss_fake
        d_loss.backward()
        d_optimizer.step()
        
        # Train Generator
        g_optimizer.zero_grad()
        output_fake = discriminator(fake_data)
        g_loss = criterion(output_fake, label_real)
        g_loss.backward()
        g_optimizer.step()
`,
        style_transfer: `
import torch
import torch.nn as nn
import torch.optim as optim

class Generator(nn.Module):
    def __init__(self, input_channels, output_channels):
        super(Generator, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(input_channels, 64, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 256, 3, padding=1),
            nn.ReLU()
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(256, 128, 3, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, 3, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(64, output_channels, 3, padding=1),
            nn.Tanh()
        )
    
    def forward(self, x):
        x = self.encoder(x)
        return self.decoder(x)

class Discriminator(nn.Module):
    def __init__(self, input_channels):
        super(Discriminator, self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(input_channels, 64, 3, padding=1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(128, 256, 3, padding=1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(256, 1, 3, padding=1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        return self.model(x)

# Training setup
generator = Generator(input_channels=3, output_channels=3)
discriminator = Discriminator(input_channels=3)
g_optimizer = optim.Adam(generator.parameters(), lr=0.0002, betas=(0.5, 0.999))
d_optimizer = optim.Adam(discriminator.parameters(), lr=0.0002, betas=(0.5, 0.999))
criterion = nn.BCELoss()

# Training loop
for epoch in range(num_epochs):
    for content_images, style_images in train_loader:
        batch_size = content_images.size(0)
        
        # Train Discriminator
        d_optimizer.zero_grad()
        label_real = torch.ones(batch_size, 1)
        label_fake = torch.zeros(batch_size, 1)
        
        output_real = discriminator(style_images)
        d_loss_real = criterion(output_real, label_real)
        
        fake_images = generator(content_images)
        output_fake = discriminator(fake_images.detach())
        d_loss_fake = criterion(output_fake, label_fake)
        
        d_loss = d_loss_real + d_loss_fake
        d_loss.backward()
        d_optimizer.step()
        
        # Train Generator
        g_optimizer.zero_grad()
        output_fake = discriminator(fake_images)
        g_loss = criterion(output_fake, label_real)
        g_loss.backward()
        g_optimizer.step()
`
    },
    
    pinn: {
        differential: `
import torch
import torch.nn as nn
import torch.optim as optim

class PINN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(PINN, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, output_size)
        )
    
    def forward(self, x):
        return self.network(x)
    
    def compute_derivatives(self, x):
        x.requires_grad_(True)
        y = self.forward(x)
        dy_dx = torch.autograd.grad(y, x, 
                                  grad_outputs=torch.ones_like(y),
                                  create_graph=True)[0]
        return y, dy_dx

# Training setup
model = PINN(input_size=1, hidden_size=20, output_size=1)
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop
for epoch in range(num_epochs):
    optimizer.zero_grad()
    
    # Physics loss (PDE)
    x_pde = torch.linspace(0, 1, 100).reshape(-1, 1)
    y, dy_dx = model.compute_derivatives(x_pde)
    pde_loss = torch.mean((dy_dx - y)**2)
    
    # Boundary condition loss
    x_bc = torch.tensor([[0.0], [1.0]])
    y_bc = model(x_bc)
    bc_loss = torch.mean((y_bc - torch.tensor([[0.0], [1.0]]))**2)
    
    # Total loss
    loss = pde_loss + bc_loss
    loss.backward()
    optimizer.step()
`,
        simulation: `
import torch
import torch.nn as nn
import torch.optim as optim

class PINN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(PINN, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, output_size)
        )
    
    def forward(self, x, t):
        inputs = torch.cat([x, t], dim=1)
        return self.network(inputs)
    
    def compute_derivatives(self, x, t):
        inputs = torch.cat([x, t], dim=1)
        inputs.requires_grad_(True)
        u = self.forward(x, t)
        
        du_dt = torch.autograd.grad(u, t, 
                                  grad_outputs=torch.ones_like(u),
                                  create_graph=True)[0]
        du_dx = torch.autograd.grad(u, x, 
                                  grad_outputs=torch.ones_like(u),
                                  create_graph=True)[0]
        d2u_dx2 = torch.autograd.grad(du_dx, x, 
                                    grad_outputs=torch.ones_like(du_dx),
                                    create_graph=True)[0]
        return u, du_dt, du_dx, d2u_dx2

# Training setup
model = PINN(input_size=2, hidden_size=20, output_size=1)
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop
for epoch in range(num_epochs):
    optimizer.zero_grad()
    
    # Physics loss (PDE)
    x = torch.linspace(0, 1, 50).reshape(-1, 1)
    t = torch.linspace(0, 1, 50).reshape(-1, 1)
    X, T = torch.meshgrid(x, t)
    x_flat = X.reshape(-1, 1)
    t_flat = T.reshape(-1, 1)
    
    u, du_dt, du_dx, d2u_dx2 = model.compute_derivatives(x_flat, t_flat)
    pde_loss = torch.mean((du_dt - d2u_dx2)**2)
    
    # Initial condition loss
    t0 = torch.zeros_like(x)
    u0 = model(x, t0)
    ic_loss = torch.mean((u0 - torch.sin(torch.pi * x))**2)
    
    # Boundary condition loss
    x0 = torch.zeros_like(t)
    x1 = torch.ones_like(t)
    u0 = model(x0, t)
    u1 = model(x1, t)
    bc_loss = torch.mean(u0**2 + u1**2)
    
    # Total loss
    loss = pde_loss + ic_loss + bc_loss
    loss.backward()
    optimizer.step()
`
    }
}; 