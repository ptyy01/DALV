import torch
import torch.nn as nn 
import my_clip.clip as m_clip
# import loralib as lora
import my_loralib as lora
import clip

class MYCLIP(nn.Module):
    def __init__(self, args):
        super(MYCLIP, self).__init__()

        self.clip_model, _, _ = m_clip.load(name=args.CLIP_Arch, device=args.device, args=args, is_train=True)
        self.clip_model.float()
        lora.mark_only_lora_as_trainable(self.clip_model)

        # prompt_prefix = "a satellite image of a"
        prompt_prefix = "This is a satellite image of a"
        # prompt_prefix = "This is a remote sensing photo of a"
        classnames = [name.replace("_", " ") for name in args.class_list]
        self.prompts = [prompt_prefix + " " + name + "." for name in classnames]

        update_params = []
        for name, params in self.clip_model.named_parameters():
            if params.requires_grad:
                update_params.append(name)
        
        print("Parameters of CLIP to be updated:", update_params)


    def forward(self, x):
        image_features = self.clip_model.encode_image(x)
        tokenized_prompts = torch.cat([m_clip.tokenize(p) for p in self.prompts]).to(x.device)
        text_features = self.clip_model.encode_text(tokenized_prompts)
        
        return image_features, text_features
    

class MYCLIP_NoLoRA(nn.Module):
    def __init__(self, args):
        super(MYCLIP_NoLoRA, self).__init__()

        self.clip_model, _, = clip.load(name=args.CLIP_Arch, device=args.device)
        self.clip_model.float()

        prompt_prefix = "This is a satellite image of a"
        classnames = [name.replace("_", " ") for name in args.class_list]
        self.prompts = [prompt_prefix + " " + name + "." for name in classnames]
        
        update_params = []
        for name, params in self.clip_model.named_parameters():
            if params.requires_grad:
                update_params.append(name)
        
        print("Parameters of CLIP to be updated:", update_params)
        # self.text_features = clip_model.encode_text(tokenized_prompts).detach().cpu()

        # self.feature_extractor = clip_model.visual
        # self.feature_extractor = self.feature_extractor.float()
        

    def forward(self, x):
        image_features = self.clip_model.encode_image(x)
        tokenized_prompts = torch.cat([m_clip.tokenize(p) for p in self.prompts]).to(x.device)
        text_features = self.clip_model.encode_text(tokenized_prompts)
        
        return image_features, text_features