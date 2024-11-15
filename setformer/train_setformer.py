import torch
from torch.utils.data import DataLoader
import yaml
from tqdm import tqdm

from ofa.utils import WordEmbedding
from setformer.dataset import collate_fn
from setformer.setformer import SetFormer

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def train_process(model: SetFormer, model_config_dict: dict, train_loader, val_loader):

    # Log the model parameter size and configs
    print(f"Totla model parameter size: {sum(p.numel() for p in model.parameters())}")
    print(f"Model parameter size without word vectors which is FROZEN: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")
    print(f"Model configs: {model_config_dict}")
    
    # Init optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=model_config_dict['training_hps']['lr'])

    # Init cos similiarity
    cosine_similarity = torch.nn.CosineSimilarity(dim=1)

    # Move the model to the device
    model.to(device)

    # Train the model and print loss values every epoch
    for epoch in range(model_config_dict['training_hps']['epochs']):
        model.train()
        total_loss = 0
        for i, (inputs, targets) in tqdm(enumerate(train_loader), total=len(train_loader), 
                                         desc=f"Epoch {epoch+1}/{model_config_dict['training_hps']['epochs']}"):
            inputs = inputs.to(device)
            targets = targets.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = (1 - cosine_similarity(outputs, targets)).mean()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()

            if i % 2 == 0:
                print(f"Epoch {epoch+1}/{model_config_dict['training_hps']['epochs']}, Batch {i+1}/{len(train_loader)}, Loss: {loss.item()}")
        
        print(f"Epoch {epoch+1}/{model_config_dict['training_hps']['epochs']}, Loss: {total_loss / len(train_loader)}")

        # Evaluate the model on the validation set
        model.eval()
        total_loss = 0
        with torch.no_grad():
            for i, (inputs, targets) in enumerate(val_loader):
                inputs = inputs.to(device)
                targets = targets.to(device)
                outputs = model(inputs)
                loss = (1 - cosine_similarity(outputs, targets)).mean()
                total_loss += loss.item()

                if i % 2 == 0:
                    print(f"Epoch {epoch+1}/{model_config_dict['training_hps']['epochs']}, Batch {i+1}/{len(val_loader)}, Loss: {loss.item()}")

            print(f"Validation Loss: {total_loss / len(val_loader)}")

    return model

def train_setformer(setformer_config_yaml,
                    multilingual_embeddings: WordEmbedding,
                    train_set,
                    val_set):
    
    # Get the hps in yaml file path
    with open(setformer_config_yaml, 'r') as file:
        setformer_config = yaml.load(file, Loader=yaml.FullLoader)

    # Prepare dataloaders
    train_loader = DataLoader(train_set, batch_size=setformer_config['training_hps']['batch_size'], 
                              shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_set, batch_size=setformer_config['training_hps']['batch_size'], 
                            shuffle=False, collate_fn=collate_fn)

    # Load the SetFormer model
    setformer = SetFormer(emb_dim=setformer_config['model_hps']['emb_dim'],
                          num_heads=setformer_config['model_hps']['num_heads'],
                          num_layers=setformer_config['model_hps']['num_layers'],
                          dim_feedforward=setformer_config['model_hps']['dim_feedforward'],
                          output_dim=setformer_config['model_hps']['output_dim'],
                          context_size=setformer_config['model_hps']['max_context_size'],
                          dropout=setformer_config['model_hps']['dropout'],
                          word_vector_emb=multilingual_embeddings)
    
    # Train the model
    setformer = train_process(setformer, setformer_config, train_loader, val_loader)

    # Save the model checkpoint
    save_file_name = f"{setformer_config['logging']['checkpoint_dir']}/{setformer_config_yaml}_model.pth"
    torch.save(setformer.state_dict(), save_file_name)