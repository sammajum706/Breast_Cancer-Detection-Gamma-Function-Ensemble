import argparse
import sys
#sys.argv=['']
#del sys
from DataAugmentation_Split import*
from Model_Train import*
from Ensemble import*


parser=argparse.ArgumentParser()
parser.add_argument("--data_dir",type=str,default="/",help="The data directory for a particular magnfication of BreakHis dataset")
parser.add_argument("--epochs",type=int,default=50,help="The number of epochs needed for optimization")
args=parser.parse_args()


data_dir=args.data_dir
epochs=args.epochs


if __name__=="__main__":
    AugData(data_dir)
    input_folder="/Final_Split/"
    os.chdir(data_dir)
    dataloaders,testloader,class_names,num_classes,dataset_sizes=get_data(input_folder)
    
    Google=Googlenet(class_names)
    Mobile=MobileNet(class_names)
    VGG11_Mod=VGG11(class_names)
    
    name=["Google","Mobile","VGG11_Mod"]
    
    Google=train_model(Google,epochs,model_name = name[0])
    Mobile=train_model(Mobile,epochs,model_name=name[1])
    VGG11_Mod=train_model(VGG11_Mod,epochs,model_name=name[2])
    
    Model=[Google,Mobile,VGG11_Mod]
    
    for i in range(len(Model)):
        print(name[i])
        print("-"*10)
        Model[i]=Model[i].eval()
        correct = 0
        total = 0
        f = open(data_dir+"/"+name[i]+".csv",'w+',newline = '')
        writer = csv.writer(f)
        with torch.no_grad():
            num = 0
            temp_array = np.zeros((len(testloader),num_classes))
            for data in testloader:
                images, labels = data
                labels=labels.cuda()
                outputs=model(images.cuda())
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels.cuda()).sum().item()
                prob = torch.nn.functional.softmax(outputs, dim=1)
                temp_array[num] = np.asarray(prob[0].tolist()[0:num_classes])
                num+=1
        print("Accuracy of "+name[i]+" = ",100*correct/total)
        for j in range(len(testloader)):
            writer.writerow(temp_array[j].tolist())
        f.close()
        print("\n")
    p1,labels = getfile(data_dir+"/"+name[0]+".csv",input_folder)
    p2,_ = getfile(data_dir+"/"+name[1]+".csv",input_folder)
    p3,_ = getfile(data_dir+"/"+name[2]+".csv",input_folder)
    
    predictions = Gamma( num_classes ,p1,p2,p3)
    correct = np.where(predictions == labels)[0].shape[0]
    total = labels.shape[0]
    
    print("Ensemble Accuracy = ",correct/total)
    classes = []
    for i in range(p1.shape[1]):
        classes.append(str(i+1))
    metrics(labels,predictions,classes)
    plot_roc(labels,predictions)