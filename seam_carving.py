import numpy as np 
from PIL import Image
import matplotlib.pyplot as plt 

def main():
    s1 = Seam_carver("image2.jpg")
    # s1.voir_image() # affiche l'image originale
    
    s1.verifier_dim() # verifier les dimension
    # energie = s1.calcule_energie()
    image_final, imgs_intermediaires,historique_seams,image_accum_rouge = s1.multiple_seam(100,True)
    s1.affichage_final(image_final,imgs_intermediaires,100)
    plt.imshow(image_accum_rouge) # affiche l'image avec tous les seams
    plt.title(f"Image rassemblant tous les seams")
    plt.show()
    

class Seam_carver:
    def __init__(self,chemin_image):
        
        #self.image_original = image.copy()
        img = Image.open(chemin_image) 
        self.image = np.array(img)

    def voir_image(self):
        plt.imshow(self.image)
        plt.title(f"Image originale -- Taille: {self.image.shape[0]}x{self.image.shape[1]}")
        plt.axis('on')
        plt.show()
    
    def verifier_dim(self):
        print(self.image[:3, :3])
        print("\nDimensions:", self.image.shape)
        print("Hauteur:", self.image.shape[0], "pixels")
        print("Largeur:", self.image.shape[1], "pixels")
        if self.image.shape == 3:
            print("Canaux:", self.image.shape[2])  # mettre cette ligne en commentaire si l'image n'est pas en couleur
        
    def rgb_to_gris(self,image):
        return np.dot(image[..., :3], [0.299, 0.587, 0.114]) #c'est une formule standard pour convertir une image RGB en gris
        
    def calcule_energie(self,image):
        if len(image.shape) == 3:
            gris = self.rgb_to_gris(image)  # si l'image n'est pas en gris
        else:
            gris = image.astype(float)  # si elle est en gris
        dy,dx = np.gradient(gris.astype(float)) # float pour éviter les overflow
                                                # grad vertical (dy): différence entre pixels haut/bas
                                                # grad horizontal (dx): différence entre pixels gauche/droite
        energie = np.abs(dx) + np.abs(dy)       # energie est la somme des valeurs absolue des gradients
        return energie
    
    def find_verticale_seam(self,energie):
        h,w = energie.shape
        cout = energie.copy().astype(float) # tableau des coûts cumulé
        
        #gtableau pour le backtracking (d'où on vient)
        # -1 = vient de gauche, 0 = du centre, 1 = de droite
        backtrack = np.zeros((h,w),dtype=int) 
        
        # Initialisation: première ligne = énergie de départ
        # Pas besoin de modifier pour la ligne 0
        for i in range(1,h): # on commence à la ligne 1
            for j in range(w):
                gauche = cout[i-1,j-1] if j>0 else float('inf')
                milieu = cout[i-1,j]
                droite = cout[i-1,j+1] if j<w-1 else float('inf')
                cout_min= min(gauche,milieu,droite) #trouver le minimum
                cout[i,j] = energie[i,j] + cout_min # mis à jour du cout
                
                #sauvegarder d'ou on vient pour le retour(backtracking)
                if cout_min == gauche:
                    backtrack[i,j] = -1
                elif cout_min == milieu:
                    backtrack[i,j]= 0
                else:
                    backtrack[i,j] = 1
                    
        seam_fin = np.argmin(cout[-1,:]) # trouver le pixel de départ sur la derniere ligne
        # backtrack pour reconstruire le chemin complet
        seam = [seam_fin]
        for i in range(h-1,0,-1): #remonter de la derniere à la premiere ligne
            j=seam[-1]
            direction = backtrack[i,j]
            seam.append(j+direction)
            
        seam.reverse() # inverser pour avoir le chemin du haut en bas
        
        return seam,cout  
    
    def visualisation_seam(self,image, seam, color=(255, 0, 0)):
    
            # Dessine la seam en rouge sur l'image
        img_avec_seam = image.copy()
        h, w = img_avec_seam.shape[:2]
        
        for i in range(h):
            j = seam[i]
            # Assurer que j est dans les limites
            j = max(0, min(j, w-1))
            
            # Dessiner un point rouge sur la seam
            img_avec_seam[i, j] = color
        
        return img_avec_seam
    
    def supp_seam(self,image,seam):
        h,w = image.shape[:2]
        # creation de la nouvelle image
        if len(image.shape) == 3: # image en couleur
            nouvelle_image = np.zeros((h,w-1,3), dtype=image.dtype)
            for i in range(h):
                j= seam[i]
                nouvelle_image[i,:,:] = np.delete(image[i,:,:],j,axis=0)
        else: #image en gris
            nouvelle_image= np.zeros((h,w-1),dtype=image.dtype)
            for i in range(h):
                j = seam[i]
                nouvelle_image[i,:] = np.delete(image[i,:],j)
        
        return nouvelle_image

    def multiple_seam(self,k_global,seams_intermediaire = False): # si False alors la sauvegarde des seams intermediaire n'auras pas lieu
        img_courante = self.image.copy() #image courante
        imgs_intermediaires = [] # liste des images avec seams 
        historique_seams = [] #un historique des jointures
        
        
        if seams_intermediaire:
            image_accum_rouge = self.image.copy()  # image_accum_rouge va contenir tous les seams sur une même image
        else:
            image_accum_rouge = None    
            
        print(f"\nEn train de réduire l'image....")
        
        #calcule de l'energie
        for k in range(k_global):
            energie = self.calcule_energie(img_courante) # on calcule l'énergie
            seam,cout = self.find_verticale_seam(energie)  # recherche de la seam maximale
            historique_seams.append(seam) 
            
            if seams_intermediaire and (k<5 or k % 10 ==0 or k==k_global-1): 
                img_avec_seam = self.visualisation_seam(img_courante,seam,color=[255,0,0]) 
                imgs_intermediaires.append((k,img_avec_seam.copy()))  # on sauvegarde l'image avec seam dans la liste
        
            # Superposer cette seam en rouge sur l'image résultat
            if seams_intermediaire and image_accum_rouge is not None:
                h, w = image_accum_rouge.shape[:2]
                
                # Approximation simple : on décale un peu vers la droite
                # pour tenir compte des suppressions précédentes
                decalage = min(k, 50)  # Décale jusqu'à 50 pixels max
                
                for i in range(min(len(seam), h)):
                    j_original = seam[i] if i < len(seam) else seam[-1]
                    j_original = max(0, min(j_original + decalage, w - 1))
                    
                    # Dessiner en rouge
                    image_accum_rouge[i, j_original] = [255, 0, 0]
                        
            # suppression de la seam             
            img_courante = self.supp_seam(img_courante,seam) 
            if k==0 or (k+1)%10 == 0 or k == k_global-1:
                print(f"k={k}") # voir la progression de k
                
        print("Processus terminé !")
        print(f"Avant réduction, les dimensions étaient: {self.image.shape[0]} x {self.image.shape[1]}")
        print(f"Après réduction, les dimensions sont: {img_courante.shape[0]} x {img_courante.shape[1]}")
            
        return img_courante, imgs_intermediaires,historique_seams,image_accum_rouge
    
    def affichage_final(self,resulat,intermediaire,k):
        plt.figure(figsize=(5,5))
        plt.imshow(self.image) # 1ere grille ---image originale
        plt.title(f"Image originale --- {self.image.shape[0]} x {self.image.shape[1]}")
        plt.axis('on')
        plt.show()
           
        
        for k,img in intermediaire:
            plt.figure(figsize=(5,5))
            plt.imshow(img)
            plt.title(f"Image après {k+1} seams (k={k}) !")
            plt.axis('off')
            plt.show(block=False)
            plt.pause(0.5)
                
        plt.figure(figsize=(5,5))
        plt.imshow(resulat) #derniere grille ---image finale
        plt.title(f"Image finale --- {resulat.shape[0]} x {resulat.shape[1]} ")
        plt.axis('on')
        plt.show()
        

if __name__ == "__main__":
    main()

    
    
    


    