
"""
@author: Juan BENCOSME
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import host_subplot
import mpl_toolkits.axisartist as AA

"PROJET: SUJET B-PARTIE 2, PROF. CORINE LACOUR"
"ETHANE (1) Y N-HEPTANE(2)"



###############################################################################
#DONNÉES DU PROJET
###############################################################################

def proj(Temp,Tmax=0):
   
    T= Temp                # Kelvin, Temperatura dada
    P= 15                  # Bar, Presion dada.
    Ten=[184.6,371.6]      # Kelvin, Temperatura Normal de EBullición
    Pc=[48.8,27.4]         # Pa, Presion critica
    Tc=[305.4,540.3]       # Kelvin, Presion critica
    omega=[0.427480,0.086640]
    w=[0.099,0.349]          # FACTEUR ACENTRIQUE ETHANE (0) et 
                             # FACTEUR ACENTRIQUE N-HEPTANE (1)
                             
    Kij=0.0041           # COEFFICIENT D'INTERACTIONS BINAIRES    
    R=8.314              # CONSTANT DU GAZ PARFAIT Joule/Mol*Kelvin
##############################################################################
#PROGRAM STARTS HERE
##############################################################################                 
    "VECTEURS DE VARIABLES"
    
    alfa= np.zeros(2)                      # CREATION DE TABLEAUX REMPLIS DE ZEROS.
    Psat=np.zeros(2)
    Prsat=np.zeros(2)
    Pr=np.zeros(2)
    Tr=np.zeros(2)
    K1=np.zeros(2)
    
    m=np.zeros(2)
    A=np.zeros(2)
    B=np.zeros(2)
    
    "DONNÉES D'ENTHALPIE et ENTROPIE "
    
    CpETh=[12.58*4.1868, 15.68*4.1868]      # CP ETHANE(1) Joule/Mol*K
    H=[6119.37240211657 , 19303.3052334930]
    CpHept= [39.67*4.1868, 50.42*4.1868]    # CP N-HEPTANE(2) Joule/Mol*K
    SETh=[310.1]                            # ENTROPIE DONNÉ DANS L'EXERCISE PAR L'ETHANE
    SHept= [427.9]                          # ENTROPIE DONNÉ DANS L'EXERCISE PAR LE N-HEPTANE
    "TEMPÉRATURE RÉDUITE"
    Tr[0]= T/Tc[0]
    Tr[1]= T/Tc[1]
    
    "PRESSION RÉDUITE"
    Pr[0]= P/Pc[0]
    Pr[1]= P/Pc[1]
    
    "PRESSION DE SATURATION ET FACTEUR ACENTRIQUE"
    Psat[0]=np.exp((Ten[0]*Tc[0]*np.log(Pc[0]))/(Tc[0]-Ten[0])*(1/Ten[0]-1/T)) # PRESSION SATURATION ETHANE (1)
    Psat[1]=np.exp((Ten[1]*Tc[1]*np.log(Pc[1]))/(Tc[1]-Ten[1])*(1/Ten[1]-1/T)) # PRESSION SATURATION N-HEPTHANE (2)
 
    Prsat[0]=Psat[0]/Pc[0]                    # PRESSION RÉDUITE DE SATURATION ETHANE (1)
    Prsat[1]=Psat[1]/Pc[1]                    # PRESSION RÉDUITE DE SATURATION N-HEPTANE (2)
    
    
    m[0]=0.480 + 1.574*w[0]-0.176*(w[0])**(2) # VALEUR DE M1 POUR L'ETHANE (1)
    m[1]=0.480 + 1.574*w[1]-0.176*(w[1])**(2) # VALEUR DE M2 POUR N-HEPTANE (2)
   
    "CALCULS DES COEFFICIENTS A1, A2, B1, B2"
    alfa[0] = (1+ m[0]*(1-(Tr[0])**(1/2)))**(2) # VALEUR DE ALFA{TR1} POUR L'ETHANE (1)
    alfa[1] = (1+ m[1]*(1-(Tr[1])**(1/2)))**(2) # VALEUR DE ALFA{TR2} POUR N-HEPTANE (2)
   
    A[0]=omega[0]*alfa[0]*(Pr[0]/(Tr[0])**(2)) # VALEUR DE A1 POUR L'ETHANE (1)
    A[1]=omega[0]*alfa[1]*(Pr[1]/(Tr[1])**(2)) # VALEUR DE A2 POUR N-HEPTANE (2)
    
    B[0]=omega[1]*(Pr[0]/(Tr[0])) # VALEUR DE B1 POUR L'ETHANE (1)
    B[1]=omega[1]*(Pr[1]/(Tr[1])) # VALEUR DE B2 POUR N-HEPTANE (2)
        
    A12= (1-Kij)*((A[0]*A[1])**(1/2)) #COEFFICIENTS D'INTERACTIONS BINAIRES
       
    "CONCENTRATIONS D'INITIALISATION -ETHANE ET HEPTANE"
    K1[0]=Psat[0]/P
    K1[1]=Psat[1]/P
    aa1=np.array([[K1[0],0,-1,0],[0,K1[1],0,-1],[1,1,0,0],[0,0,1,1]])
    b1=np.array([0,0,1,1])
    XX1=np.linalg.solve(aa1,b1)
    x1=XX1[0]
    x2=XX1[1]
    y1=XX1[2]
    y2=XX1[3]
    XI=[x1,x2]   
    YI=[y1,y2]
    
    # PROCEDURE DE L'ETAPE SUIVANTE DU CODE:
    # 1. RESOLUTION D'EQUATION POLYNOMIALE EN Z POUR TROUVER ZL ET ZV (ZL ETANT LA RACINE LA PLUS PETITE ET ZV ETANT LA RACINE LA PLUS GRAND) 
    # 2. CALCUL DE LA FUGACITE FL , FV UTILISANT LES EQUATIONS D'ETAT PRECEDENTES.
    # 3. CALCUL DE CONSTANTE D'EQUILIBRE KL , KV 6.
    # 4. CALCUL DE LA SOMME DE (Xi-Yi) ET TESTER SI LA SOMME EST EGALE A ZERO ALORS ARRETER LE CALCUL.
    # 5. CALCUL DU ENTHALPIE ET D'AUTRES GRANDEURS THERMODYNAMIQUES.
    # 5. RESULTATS.
    err = 1
    i = 0
    while  err >= 0 and i < 150 :
        i= i + 1
        X=[x1,x2]    
        Al= ((X[0]**(2))*A[0] + 2*X[0]*X[1]*A12 + (X[1]**(2))*A[1])
        Bl=X[0]*B[0]+X[1]*B[1]
        coeff= [1, -1,(Al-Bl-(Bl)**(2)), -Al*Bl]
        JJ21=np.roots(coeff)
        Jl=(sorted(JJ21)[0].real) #ZL LA RACINE LA PLUS PETITE 
        
          
        phi1L=np.exp((B[0]/Bl*(Jl-1))-np.log(Jl-Bl)-(Al/Bl)*((2/Al)*(A[0]*X[0]+X[1]*A12)-B[0]/Bl)*(np.log(1+Bl/Jl)))
        phi2L=np.exp((B[1]/Bl*(Jl-1))-np.log(Jl-Bl)-(Al/Bl)*((2/Al)*(A[1]*X[1]+X[0]*A12)-B[1]/Bl)*(np.log(1+Bl/Jl)))
     
        Y=[y1,y2]
        Av = (Y[0]**(2))*A[0] + 2*Y[0]*Y[1]*A12 + (Y[1]**(2))*A[1]
        Bv = Y[0]*B[0]+Y[1]*B[1]
        coeff1= [1,-1,(Av-Bv-(Bv)**(2)), -Av*Bv]
        JJR23=np.roots(coeff1)
        Jv=(sorted(JJR23)[2].real) #ZV ETANT LA RACINE LA PLUS GRAND
      
      
        phi1V=np.exp(B[0]/Bv*(Jv-1)-np.log(Jv-Bv)-(Av/Bv)*((2/Av)*(Y[0]*A[0]+Y[1]*A12)-B[0]/Bv)*np.log(1+ Bv/Jv))
        phi2V=np.exp(B[1]/Bv*(Jv-1)-np.log(Jv-Bv)-(Av/Bv)*((2/Av)*(Y[1]*A[1]+Y[0]*A12)-B[1]/Bv)*np.log(1+ Bv/Jv))
       
        
        PHIL=[phi1L,phi2L]
        PHIV=[phi1V,phi2V]
        K=np.zeros(2)
        K[0]=PHIL[0]/PHIV[0]
        K[1]=PHIL[1]/PHIV[1]
        
        aa=np.array([[K[0],0,-1,0],[0,K[1],0,-1],[1,1,0,0],[0,0,1,1]])
        b=np.array([0,0,1,1])
        XX=np.linalg.solve(aa,b)
        err=abs((XX[0]+XX[1]-XX[2]-XX[3]))
        x2=XX[1]
        y1=XX[2]
        y2=XX[3]
  
        
        "CALCULS D'ENTHALPIE DE MELANGE"
        
        corchetev=1+ (1/Av)*(((Y[0])*(m[0]*((Tr[0]/alfa[0])**(1/2)))*(Y[0]*A[0]+Y[1]*A12))  + ((Y[1])*(m[1]*((Tr[1]/alfa[1])**(1/2)))*(Y[0]*A12+Y[1]*A[1])))
        corchetel=1+ (1/Al)*(((Y[0])*(m[0]*((Tr[0]/alfa[0])**(1/2)))*(Y[0]*A[0]+Y[1]*A12))  + ((Y[1])*(m[1]*((Tr[1]/alfa[1])**(1/2)))*(Y[0]*A12+Y[1]*A[1])))
        deltahv=R*T*((Jv-1)-(Av/Bv*np.log(1+Bv/Jv))*corchetev)
        deltahl=R*T*((Jl-1)-(Al/Bl*np.log(1+Bl/Jl))*corchetel)
        hvetoile=(H[0])
        hletoile=(H[1])
        hvetoilem=hvetoile*XX[2] + H[1]*XX[3]
        EnthalphieV= deltahv + hvetoilem 
        hletoilem=hletoile*XX[0] + hvetoile*XX[1]
        EnthalphieL= deltahl+ hletoilem
        
        "CALCULS D'ENTROPIE DE MELANGE"
        
        deltasv=R*(np.log(Jv*(1-Bv/Jv)))-Av/Bv*m[0]*((Tr[0]/alfa[0])**(1/2))*np.log(1+Bv/Jv)
        deltasl=R*(np.log(Jl*(1-Bl/Jl)))-Al/Bl*m[1]*((Tr[1]/alfa[1])**(1/2))*np.log(1+Bl/Jl)
        svetoile=SETh[0]- R*np.log(P/101325) + CpETh[1]*np.log(T/298)
        svetoilem=svetoile*XX[2] + svetoile*XX[3]
        EntropieV= deltasv+ svetoilem
                  
        sletoile=SHept[0]- R*np.log(P/101325) + CpHept[1]*np.log(T/298)
        sletoilem=sletoile*XX[0] + svetoile*XX[1]
        EntropieL=deltasl+sletoilem
             
        
    if Tmax==T :
        "RESULTATS"
        
        print("\nNombre d'iteration =",round(i))
        print("\nError d'iteration =",round(err))
        
        print("\n   DONNÉES CORPS PUR:")
        
        print("\nCOEFFICIENT D'INTERACTIONS BINAIRES:","Kij=",(Kij))
        print("\nCOEFFICIENT D'INTERACTIONS MELANGE:","Aij(A12)=",round(A12,3))
        print( )
        
        print("\nValeurs d'Alpha':")
        print( "ETHANE(1)=",round(alfa[0],3), ",","HEPTANE(2)=", round(alfa[1],3))
        
        print("\nPression de Saturation:")
        print( "ETHANE(1)=",round(Psat[0],3), ",","HEPTANE(2)=", round(Psat[1],3))
        
        print("\nValeur de m:")
        print( "ETHANE(1)=",round(m[0],3), ",","HEPTANE(2)=", round(m[1],3))
        
        print("\nValeur de A:")
        print( "ETHANE(1)=",round(A[0],3), ",","HEPTANE(2)=", round(A[1],3))
        
        print("\nValeur de B:")
        print( "ETHANE(1)=",round(B[0],3), ",","HEPTANE(2)=", round(B[1],3))
        print("\n  COEFFICIENTS DU POLYNOME Z :")    
        print("\nCoefficients du polynome Z (GAZ-ETHANE):")          
        print("A0=",round(coeff1[0],3), ",","A1=",round(coeff1[1],3) , ",","A2=",round(coeff1[2],3), ",","A3=",round(coeff1[3],3))
        
        print("\nCoefficients du polynome Z (LIQUIDE-HEPTANE):")          
        print("A0=",round(coeff[0],3), ",","A1=",round(coeff[1],3) , ",","A2=",round(coeff[2],3), ",","A3=",round(coeff[3],3))
        
        
        print("\n  RACINES DU POLYNOME Z:")        
        print("\nRacines de l'equation en Z-Gaz (ETHANE):")
        print( "Racine Plus Petite (ETHANE)=",round(sorted(JJR23)[0].real,3), ",","Racine Plus Grand (ETHANE)=", round(sorted(JJR23)[2].real,3))
        print("\nRacines de l'equation en Z-Liquide (HEPTANE):")        
        print( "Racine Plus Petite (HEPTANE)=",round(sorted(JJ21)[0].real,3), ",","Racine Plus Grand (HEPTANE)=", round(sorted(JJ21)[2].real,3))
                    
        print("\n   DONNÉES DU MELANGE:")
        
        print("\nTitres d'initialisation X (Liquide):")
        print("X1 =", round(XI[0],3),",","X2 =", round(XI[1],3))
        print("\nTitres d'initialisation Y (Gaz):")
        print("Y1 =", round(YI[0],3),",","Y2 =", round(YI[1],3))
        
        print("\nValeur de A (Melange):")
        print( "ETHANE(1)=",round(Av,3), ",","HEPTANE(2)=", round(Al,3))
        
        print("\nValeur de B (Melange):")
        print( "ETHANE(1)=",round(Bv,3), ",","HEPTANE(2)=", round(Bl,3))
        
        print("\nConstantes d'Equilibre d'initialisation:")
        print("K1(Ethane)=", round(K1[0],3),",", "K2(Heptane)=",round(K1[1],3))
                     
        print("\nFugacité (1) du Vapeur:")
        print( "ETHANE(1)=",round(PHIV[0],3), ",","HEPTANE(2)=", round(PHIV[1],3))
        
        print("\nFugacité (2) du Liquide:")
        print("ETHANE(1)=",round(PHIL[0],3), ",","HEPTANE(2)=", round(PHIL[1],3))
        
        print("\nTitres  X (Liquide):")
        print("X1 =", round(XX[0],3),",","X2 =", round(XX[1],3))
        print("\nTitres  Y (Gaz):")
        print( "Y1 =", round(XX[2],3),",","Y2 =",round(XX[3],3))
        
        print("\nConstantes d'Equilibre:")
        print("K1=", round(K[0],3),",", "K2=",round(K[1],3))
       
        print("\nÉcart enthalpie-melange molaire vapeur et liquide (KJ/mol):")
        print("HV-HV*=", round(deltahv/1000,3),",", "HL-HL*=", round(deltahl/1000,3))
        
        print("\nÉcart enthalpie etoile-melange molaire vapeur et liquide (KJ/mol):")
        print("HV*=", round(hvetoilem/1000,3),",", "HL*=", round(hletoilem /1000,3))
        
        print("\nEnthalpie melange  dans le vapeur (KJ/mol):")
        print("HV=", round(EnthalphieV/1000,3),",", "HL=", round(EnthalphieL/1000,3))
        
        print("\nÉcart entropie-melange molaire vapeur et liquide (KJ/mol*K):")
        print("SV-SV*=", round(deltasv/1000,3),",", "SV-SL*=", round(deltasl/1000,3))
        
        print("\nÉcart entropie-melange molaire vapeur et liquide (KJ/mol*K):")
        print("SV*=", round(svetoilem/1000,3),",", "SL*=", round(sletoilem/1000,3))
        
        print("\nEntropie-melange molaire dans le vapeur  (KJ/mol*K):")
        print("SV=", round(EntropieV/1000,3),",", "SL=", round(EntropieL/1000,3))
             
    ListTemp1=[x2,y2,deltahv,deltahl]
    return  ListTemp1

"EVALUATIONS DE FONCTION POUR CONSTRUIRE LE DIAGRAMME DE PHASE "
npoints=50
T=np.zeros(npoints)  
Tmax= 400
T0=298
Tmin=T0
ListResult=[]

for j in range (npoints):
    T[j] = Tmin + (Tmax-Tmin)*j/(npoints-1)
    
    
    proj(T[j], Tmax)
    ListResult.append(proj(T[j]))

m1=np.asarray(ListResult)
XX2=m1.T[0]
YY2=m1.T[1]
EVF=m1.T[2]
ELF=m1.T[3]

"GRAPHIQUE PHASE ETHANE-HEPTANE"

host = host_subplot(111, axes_class=AA.Axes)
plt.subplots_adjust(top=1.0, right=0.95)

par1 = host.twiny()
par2 = host.twiny()

offset = 0
new_fixed_axis = par2.get_grid_helper().new_fixed_axis
par2.axis["bottom"] = new_fixed_axis(loc="bottom",
                                    axes=par2,
                                    offset=(offset, -40))



par1.axis["bottom"].toggle(all=True)
par2.axis["bottom"].toggle(all=True)


host.set_ylim(280, 420)

host.set_xlabel("Fraction Liquide-Ethane")
host.set_ylabel("Temperature (K)")
par2.set_xlabel("Fraction Vapeur-Ethane")


p2, = par1.plot(1-XX2,T,'b',label='Liquide',markersize = 10)
p3, = par2.plot(1-YY2,T,'r--',label='Vapeur',markersize = 10)

host.legend(loc="best")
plt.title("Diagramme de phase Ethane-Heptane",loc="center", color="k") 

par1.axis["bottom"].label.set_color(p2.get_color())
par2.axis["bottom"].label.set_color(p3.get_color())

plt.grid(color='k', linestyle='--', linewidth=0.5)
plt.savefig('Phase Ethane-Heptane.png', bbox_inches='tight', dpi=2500)
plt.show()
