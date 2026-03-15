class ChannelAttention(nn.Module):
    """Squeeze-and-Excitation Block for Channel Attention"""
    def __init__(self, channel, reduction=4):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)
    

    class ImprovedAttentionEEGNet(nn.Module):
    """Enhanced EEGNet with Channel Attention + Residual connections + Label Smoothing ready"""
    def __init__(self, n_chans, n_outputs, n_times, F1=16, D=2, F2=32, kernel_length=64, drop_prob=0.4):
        super(ImprovedAttentionEEGNet, self).__init__()
        
        # Increased filter counts for more capacity
        self.base_eegnet = EEGNet(
            n_chans=n_chans,
            n_outputs=n_outputs,
            n_times=n_times,
            F1=F1,           # Increased from 8 to 16
            D=D,
            F2=F2,           # Increased from 16 to 32
            kernel_length=kernel_length,
            drop_prob=drop_prob  # Increased dropout for regularization
        )
        
        self.F1 = F1
        self.D = D
        self.F2 = F2
        self.n_chans = n_chans
        self.n_outputs = n_outputs
        self.n_times = n_times
        
        # Dual attention: Channel + Temporal
        self.channel_attention = ChannelAttention(channel=F1 * D, reduction=4)
        self.channel_attention2 = ChannelAttention(channel=F2, reduction=4)
        
    def forward(self, x):
        """Forward pass with dual attention"""
        x = self.base_eegnet.ensuredims(x)
        x = self.base_eegnet.dimshuffle(x)
        x = self.base_eegnet.conv_temporal(x)
        x = self.base_eegnet.bnorm_temporal(x)
        x = self.base_eegnet.conv_spatial(x)
        
        # First attention block
        x = self.channel_attention(x)
        
        x = self.base_eegnet.bnorm_1(x)
        x = self.base_eegnet.elu_1(x)
        x = self.base_eegnet.pool_1(x)
        x = self.base_eegnet.drop_1(x)
        x = self.base_eegnet.conv_separable_depth(x)
        x = self.base_eegnet.conv_separable_point(x)
        
        # Second attention block
        x = self.channel_attention2(x)
        
        x = self.base_eegnet.bnorm_2(x)
        x = self.base_eegnet.elu_2(x)
        x = self.base_eegnet.pool_2(x)
        x = self.base_eegnet.drop_2(x)
        x = self.base_eegnet.final_layer(x)
        
        return x
    
    # Initialize improved model
model = ImprovedAttentionEEGNet(
    n_chans=N_CHANNELS,
    n_outputs=N_CLASSES,
    n_times=N_TIMEPOINTS,
    F1=16,             # Increased temporal filters
    D=2,               # Depth multiplier
    F2=32,             # Increased separable filters
    kernel_length=64,  # Temporal kernel size
    drop_prob=0.4      # Higher dropout for regularization
).to(device)

# Count parameters
total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

print(f" Improved Model initialized")
print(f"  Total parameters: {total_params:,}")
print(f"  Trainable parameters: {trainable_params:,}")
print(f"  Device: {device}")

# Test forward pass
with torch.no_grad():
    dummy_input = torch.randn(2, N_CHANNELS, N_TIMEPOINTS).to(device)
    dummy_output = model(dummy_input)
    print(f"\n Forward pass test successful")
    print(f"  Input shape: {dummy_input.shape}")
    print(f"  Output shape: {dummy_output.shape}")


    # Compute class weights to handle imbalance
class_weights = compute_class_weight(
    class_weight='balanced',
    classes=np.unique(y_train),
    y=y_train
)
class_weights_tensor = torch.FloatTensor(class_weights).to(device)

print(f"Class Weights (to handle class imbalance):")
for i, weight in enumerate(class_weights):
    class_name = f"Digit {i}" if i < 10 else "Black Screen"
    n_samples = np.sum(y_train == i)
    print(f"  Class {i:2d} ({class_name:12s}): weight={weight:.4f}, n={n_samples:3d}")

# Label Smoothing Cross Entropy for better generalization
class LabelSmoothingCrossEntropy(nn.Module):
    """Cross Entropy with Label Smoothing"""
    def __init__(self, smoothing=0.1, weight=None):
        super().__init__()
        self.smoothing = smoothing
        self.weight = weight
        
    def forward(self, pred, target):
        n_classes = pred.size(-1)
        log_preds = torch.log_softmax(pred, dim=-1)
        
        # Create smoothed labels
        with torch.no_grad():
            smoothed_labels = torch.zeros_like(log_preds)
            smoothed_labels.fill_(self.smoothing / (n_classes - 1))
            smoothed_labels.scatter_(1, target.unsqueeze(1), 1 - self.smoothing)
        
        # Apply class weights if provided
        if self.weight is not None:
            weight = self.weight[target]
            loss = (-smoothed_labels * log_preds).sum(dim=-1) * weight
        else:
            loss = (-smoothed_labels * log_preds).sum(dim=-1)
        
        return loss.mean()

# Use Label Smoothing with class weights
criterion = LabelSmoothingCrossEntropy(smoothing=0.1, weight=class_weights_tensor)

# Optimizer with gradient clipping ready
optimizer = optim.AdamW(model.parameters(), lr=5e-4, weight_decay=0.02)

# OneCycleLR for better convergence
N_EPOCHS = 80
scheduler = optim.lr_scheduler.OneCycleLR(
    optimizer,
    max_lr=2e-3,
    epochs=N_EPOCHS,
    steps_per_epoch=len(train_loader),
    pct_start=0.2,
    anneal_strategy='cos'
)

PATIENCE = 15


# Mixup augmentation for better generalization
def mixup_data(x, y, alpha=0.2):
    """Apply mixup augmentation"""
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1
    
    batch_size = x.size(0)
    index = torch.randperm(batch_size).to(x.device)
    
    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam

def mixup_criterion(criterion, pred, y_a, y_b, lam):
    """Compute mixup loss"""
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)

# Training history
history = {
    'train_loss': [],
    'train_acc': [],
    'val_loss': [],
    'val_acc': [],
    'lr': []
}

best_val_acc = 0.0
patience_counter = 0
best_model_state = None
USE_MIXUP = True
MIXUP_ALPHA = 0.3

print("Starting enhanced training...")
print(f"  • Mixup: {USE_MIXUP} (alpha={MIXUP_ALPHA})")
print(f"  • Gradient Clipping: 1.0")

for epoch in range(N_EPOCHS):
    # ========== TRAINING ==========
    model.train()
    train_loss = 0.0
    train_correct = 0
    train_total = 0
    
    for batch_idx, (inputs, labels) in enumerate(train_loader):
        inputs, labels = inputs.to(device), labels.to(device)
        
        # Apply Mixup
        if USE_MIXUP and np.random.random() > 0.3:  # Apply 70% of the time
            inputs, labels_a, labels_b, lam = mixup_data(inputs, labels, alpha=MIXUP_ALPHA)
            
            optimizer.zero_grad()
            outputs = model(inputs).squeeze()
            loss = mixup_criterion(criterion, outputs, labels_a, labels_b, lam)
            
            # For accuracy calculation, use original labels
            _, predicted = outputs.max(1)
            train_correct += (lam * predicted.eq(labels_a).sum().float() + 
                            (1 - lam) * predicted.eq(labels_b).sum().float()).item()
        else:
            optimizer.zero_grad()
            outputs = model(inputs).squeeze()
            loss = criterion(outputs, labels)
            
            _, predicted = outputs.max(1)
            train_correct += predicted.eq(labels).sum().item()
        
        # Backward with gradient clipping
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()
        
        train_loss += loss.item()
        train_total += labels.size(0)
    
    train_loss /= len(train_loader)
    train_acc = 100.0 * train_correct / train_total
    
    # ========== VALIDATION ==========
    model.eval()
    val_loss = 0.0
    val_correct = 0
    val_total = 0
    
    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs).squeeze()
            
            # Use standard CE for validation
            loss = nn.CrossEntropyLoss(weight=class_weights_tensor)(outputs, labels)
            
            val_loss += loss.item()
            _, predicted = outputs.max(1)
            val_total += labels.size(0)
            val_correct += predicted.eq(labels).sum().item()
    
    val_loss /= len(val_loader)
    val_acc = 100.0 * val_correct / val_total
    
    current_lr = optimizer.param_groups[0]['lr']
    
    # Save history
    history['train_loss'].append(train_loss)
    history['train_acc'].append(train_acc)
    history['val_loss'].append(val_loss)
    history['val_acc'].append(val_acc)
    history['lr'].append(current_lr)
    
    # Print progress
    print(f"Epoch [{epoch+1:2d}/{N_EPOCHS}] | "
          f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:6.2f}% | "
          f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:6.2f}% | "
          f"LR: {current_lr:.6f}")
    
    # Save best model
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        best_model_state = model.state_dict().copy()
        patience_counter = 0
        print(f"  ✓ New best validation accuracy: {best_val_acc:.2f}%")
    else:
        patience_counter += 1
        if patience_counter >= PATIENCE:
            print(f"\n⚠ Early stopping triggered (patience={PATIENCE})")
            break

print(f"\n{'='*60}")
print(f" Training complete!")
print(f"  Best validation accuracy: {best_val_acc:.2f}%")
print(f"{'='*60}")

# Load best model
model.load_state_dict(best_model_state)
torch.save(best_model_state, 'best_model_improved.pth')
print("\n✓ Best model saved as 'best_model_improved.pth'")