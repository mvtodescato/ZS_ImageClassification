import warnings
warnings.filterwarnings("ignore")
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize
from torch.utils.data import Dataset, DataLoader, random_split
from PIL import Image
from train.data_loaders import dataset_loader
import clip
import torch
from tqdm import tqdm
import numpy as np
import faiss
from sklearn import metrics
import getopt, sys
sys.path.append('./CLIP') 


target_col = 'class_label'  # Default column with labels
input_resolution = 224    # Default input resolution


# %%
torch_device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
device = "cuda" if torch.cuda.is_available() else "cpu"
device

# %%
def to_rgb(image):
    return image.convert("RGB")


# General transformation applied to all models
preprocess_image = Compose(
    [
        Resize(input_resolution, interpolation=Image.BICUBIC),
        CenterCrop(input_resolution),
        to_rgb,
        ToTensor(),
    ]
)


# %%
def torch_hub_normalization():
    # Normalization for torch hub vision models
    return Normalize(
        (0.485, 0.456, 0.406),
        (0.229, 0.224, 0.225),
    )


# %%
def clip_normalization():
    # SRC https://github.com/openai/CLIP/blob/e5347713f46ab8121aa81e610a68ea1d263b91b7/clip/clip.py#L73
    return Normalize(
        (0.48145466, 0.4578275, 0.40821073),
        (0.26862954, 0.26130258, 0.27577711),
    )



# %%
# Dataset loader
class ImagesDataset(Dataset):
    def __init__(self, df, preprocess, input_resolution):
        super().__init__()
        self.df = df
        self.preprocess = preprocess
        self.empty_image = torch.zeros(3, input_resolution, input_resolution)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        row = self.df.iloc[index]

        try:
            if 'Filename' in row:
                image = self.preprocess(Image.open(row['Filename']))
            else:
                image = self.preprocess(Image.fromarray(row['Image']))    
        except:
            print("error")
            image = self.empty_image

        return image, row[target_col]

def add_vector_to_index(embedding, index):
    #convert embedding to numpy
    vector = embedding.detach().cpu().numpy()
    #Convert to float32 numpy
    vector = np.float32(vector)
    #Normalize vector
    faiss.normalize_L2(vector)
    #Add to index
    index.add(vector)

def image_classification(dataset,backbone):
    #Set the dataset, load the features
    dataset_name = dataset
    df_all, LABELS_MAP, SPLITS = dataset_loader(dataset_name)
    
    SPLIT_SIZE = len(LABELS_MAP)
    class BaseNeuralNetwork(nn.Module):
        def __init__(self):
            super().__init__()
            self.network = nn.Sequential(
                nn.Linear(1024, SPLIT_SIZE)     
            )

        def forward(self, x):
            return self.network(x)

    with open('D:/features_models/X_' + dataset_name + '_vitg14.npy','rb') as f:
        X = np.load(f)

    with open('y_' + dataset_name + '.npy','rb') as f:      
        y = np.load(f)


    image_normalization = torch_hub_normalization()
    preprocess = Compose([preprocess_image, image_normalization])
    ds = ImagesDataset(df_all, preprocess, input_resolution)

    classes = [(f"a photo of a {c}") for c in LABELS_MAP]
    
    model, _ = clip.load(backbone, device=device)

    if backbone == "ViT-L/14":
        emb_size = 768
        fname = 'clipl14_'
    elif backbone == "ViT-B/32":
        emb_size = 512
        fname = 'clip_'
    else:
        emb_size = 512
        fname = 'clipb16_'

    #Encode the images with CLIP
    try:
        with open('X_'+ fname + dataset_name + '.npy','rb') as f:
            X_clip = np.load(f)
    except:
        X_clip = np.empty((len(ds), emb_size), dtype=np.float32)
        i = 0
        for images, class_name in tqdm(ds):
                image = images.unsqueeze(0).to(device)

                with torch.no_grad():
                    image_features = model.encode_image(image)
                image_features = image_features.detach().cpu().numpy()

                X_clip[i] = np.float32(image_features)

                i += 1
        np.save(f'X_'+ fname + dataset_name + '.npy', X_clip)

    #Index the encoded images
    X_index = X_clip
    index = faiss.IndexFlatL2(emb_size)
    for embedding in X_index:
        
        embedding = torch.from_numpy(embedding)
        embedding = embedding.to(device)
        embedding = embedding.unsqueeze(0)

        add_vector_to_index(embedding,index)

    #Generate the ranking for each seen class
    top5list = []
    for id_class in range(len(classes)):
        text = clip.tokenize(classes[id_class])
        with torch.no_grad():
            text_features = model.encode_text(text.cuda())

        text_np = text_features.detach().cpu().numpy()
        text_np = np.float32(text_np)

        distances, indices = index.search(text_np, 100)

        top5list.append(indices[0])

        for i,v in enumerate(indices[0]):
            sim = (1/(1+distances[0][i])*100)
    
    text_inputs = torch.cat([clip.tokenize(f"a photo of a {c}") for c in LABELS_MAP]).to(device)
    # Calculate features
    with torch.no_grad():
        text_features = model.encode_text(text_inputs)
    text_features /= text_features.norm(dim=-1, keepdim=True)

    conf_list = []
    cont_class = 0
    for top100 in top5list:
        confidence = []
        for i in range(100):
            conf_sum = 0
            embedding = torch.from_numpy(X_clip[top100[i]])
            image = embedding.unsqueeze(0)
            distances, indices = index.search(image, 6)
            distances = list(distances[0])
            distances = distances[1:]
            indices = list(indices[0])
            indices = indices[1:]
            for distance, ind in list(zip(distances,indices)):
                image_features = torch.tensor(X_clip[ind], dtype=text_features.dtype, device=device).unsqueeze(0)
                image_features /= image_features.norm(dim=-1, keepdim=True)
                similarity = (100.0 * image_features @ text_features.T).softmax(dim=-1)
                values, indexs = similarity[0].topk(5)

                if indexs[0] == cont_class:
                    conf_sum += values[0] * 20
                elif indexs[1] == cont_class:
                    conf_sum += values[1] * 20
                elif indexs[2] == cont_class:
                    conf_sum += values[2] * 20
                elif indexs[3] == cont_class:
                    conf_sum += values[3] * 20
                elif indexs[4] == cont_class:
                    conf_sum += values[4] * 20
            confidence.append(conf_sum)
        print(cont_class)
        cont_class += 1
        conf_list.append(confidence)
    

    newtop5list = []
    for i in range(len(top5list)):
        indices = top5list[i]
        confidence = conf_list[i]
        sorted_indices = [x for _, x in sorted(zip(confidence, indices), key=lambda pair: pair[0], reverse=True)]
        newtop5list.append(sorted_indices)

    #Select the seed
    train_y = []
    few_shot_indices = []
    cont_class = 0
    for top5 in newtop5list:
            for indeximg in range(5):
                few_shot_indices.append(top5[indeximg])
                train_y.append(cont_class)
            cont_class += 1
        

    train_X = X[few_shot_indices]
    train_y = train_y

    fold_model = BaseNeuralNetwork()
    fold_model.to(torch_device)

    print("Classifier")
    print(fold_model)
    print()

    fold_criterion = nn.CrossEntropyLoss()

    fold_optimizer = torch.optim.Adam(fold_model.parameters(), lr=0.001) 

    EPOCHS = 100
    print("Loss", fold_criterion)
    print("Optimizer", fold_optimizer)
    print()

    
    print("Start training ...")
    for epoch in range(EPOCHS):
                running_loss = 0.0

                train_loader = DataLoader(list(zip(train_X, train_y)), batch_size=16, shuffle=True,
                            num_workers=0, pin_memory=True)

                for i, data in enumerate(train_loader, 0):

                    inputs, label_index = data

                    multilabel_values = np.zeros((len(label_index),SPLIT_SIZE)).astype(float)

                    for k, idx in enumerate(label_index):
                        multilabel_values[k][idx] = 1.0


                    tensor_multilabel_values = torch.from_numpy(multilabel_values).to(torch_device)

                    fold_optimizer.zero_grad()

                    outputs = fold_model(inputs.to(torch_device))
                    pred = outputs.cpu().argmax()

                    fold_loss = fold_criterion(outputs, tensor_multilabel_values.float())
                    fold_loss.backward()
                    fold_optimizer.step()

                    running_loss += fold_loss.item()
                    
                    if i == len(train_loader) - 1:  
                        print('[%d, %5d] Train loss: %.5f' %
                            (epoch + 1, i + 1, running_loss / len(train_loader)))
                        running_loss = 0.0


    test_X = X
    test_y = y

    corrects = 0
    fold_cm = np.zeros((len(LABELS_MAP), len(LABELS_MAP))).astype(int)
    y_pred = []
    test_predictions = []

    print("Start testing ...")

    for x_item, y_item in list(zip(test_X, test_y)):

        item_input = torch.from_numpy(x_item).to(torch_device)

        preds = fold_model(item_input)

        pred_index = preds.cpu().argmax()

        fold_cm[y_item][pred_index] += 1

        if pred_index == y_item:
            corrects += 1

        y_pred.append(pred_index)
        test_predictions.append(preds.detach().cpu().numpy().tolist())

    
    #Calculatin the metrics
    y_pred = np.array(y_pred)
    accuracy_score = corrects / len(test_y)     
    
    print(f"{corrects}/{len(test_y)} = val_acc {accuracy_score:.5f}")

    print("Classification Report:")
    print(metrics.classification_report(test_y, y_pred, target_names=LABELS_MAP))
    print()

    print(f"{corrects}/{len(test_y)} = val_acc {accuracy_score:.4f}")
    with open("results_ic_noloop.txt", "a") as text_file:
        text_file.write(str(backbone) + ' ' + str(dataset_name) + ' ' + f"{corrects}/{len(test_y)} = val_acc {accuracy_score:.4f}" + "\n")
    
def main():
    argumentList = sys.argv[1:]
    # Options
    options = "hD:B:"
    # Long options
    long_options = ["Help", "Dataset=", "Backbone="]
    dataset = ''
    backbone = ''
    try:
        # Parsing argument
        arguments, values = getopt.getopt(argumentList, options, long_options)
        
        # checking each argument
        for currentArgument, currentValue in arguments:
    
            if currentArgument in ("-h", "--help"):
                print("No Labels Needed: Zero-Shot Image Classification with Collaborative Self-Learning\n")
                print('Usage:')
                print('python ic_wloop_paper.py -h | --help')
                print("python ic_wloop_paper.py -D <dataset> -B <backbone> ")
                print('\nOptions:')
                print("-h --help    Show this screen")
                print("-D           Dataset name [datasets available below] (Its important to notice that you need to use the simple_features.py code to extract features before start this process)")
                print("-B           CLIP backbone (Options: ViT-B/16,ViT-B/32,ViT-L/14) (more details in the paper)")
                print("\nAvailable datasets (call by the name in the right): ")
                print("CIFAR10:         cifar10")
                print("CIFAR100:        cifar100")
                print("Stanford Cars:   cars")
                print("Caltech-101:     caltech")
                print("Caltech-256:     caltech256")
                print("ImageNet:        imagenet_val")
                print("Flowers:         flowers")
                print("Food-101:        food")
                print("Textures:        textures")
                print("Aircraft:        aircraft")
                print("\nMore details of the approach and the implementation in the paper")
                sys.exit(2)

            elif currentArgument in ("-D"):
                dataset = currentValue
            
            elif currentArgument in ("-B"):
                backbone = currentValue
                
        print("Dataset: ", dataset)
        print("Backbone: ", backbone)

    except UnboundLocalError:
        print("You forgot to define something")
        sys.exit(2)
    except getopt.error as err:
        print (str(err))
        sys.exit(2)

    image_classification(dataset,backbone)
    


if __name__ == "__main__":
   main()
