from turtle import distance
import torch 
import torch.nn.functional as F

class Optimization():
    def __init__(self, train_loader, device):
        self.train_loader = train_loader
        self.device = device

    def cdw_feature_distance(self, old_model, old_classifier, old_bottleneck, new_model):
        """cosine distance weight (cdw): calculate feature distance of 
           the features of a batch of data by cosine distance.
        """
        # for i in range(0, model_len):
        #     model_layer_name.append(model_layer_name_sub_1 + str(i) + model_layer_name_sub_2)
        # hard_loss_idx = hard_loss.index(max(hard_loss))
        old_model.eval()
        old_model=old_model.to(self.device)
        
        old_classifier=old_classifier.to(self.device)
        old_bottleneck = old_bottleneck.to(self.device)
        # for name, para in old_model.named_parameters():
        #     print(name, '    ', para.size())
        for index, data in enumerate(self.train_loader):

            inputs, _ = data
            inputs=inputs.to(self.device)
            with torch.no_grad():

                # for name, param  in old_model()
                old_out = old_model(inputs)
                old_out_bn = old_bottleneck(old_out)
                old_score = old_classifier(old_out_bn)
                # old_out = old_classifier(old_model(inputs))
                # print('old_out', old_out.shape)
                new_score, new_out= new_model(inputs)

                # old_out = old_classifier(old_model(inputs))
                # new_out = new_model(inputs)
            distance = 1 - torch.cosine_similarity(old_out, new_out)
            # score_distance = 1 - torch.cosine_similarity(old_score, new_score)
            # distance = 0.5 * feat_distance + 0.5 * score_distance

            # distance = 1 - torch.cosine_similarity(old_out, new_out) + 1 - torch.cosine_similarity(old_score, new_score)
            # return torch.mean(score_distance)
            return torch.mean(distance)

    def kd_generate_soft_label(self, model, data, regularization):
        """knowledge distillation (kd): generate soft labels.
        """
        # result = model(data)
        # model = model.to(self.device)
        score, result = model(data)
        if regularization:
            result = F.normalize(result, dim=1, p=2)
        return result
