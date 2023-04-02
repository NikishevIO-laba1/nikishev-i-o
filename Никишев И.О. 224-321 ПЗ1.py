# 1. Задать цвет в координатах LAB
# 2. Преобразовать цвет заданный в п.1 в цветовое пространство LCH
# 3. Преобразовать цвет заданный в п.1 в цветовое пространство RGB
# 4. Преобразовать цвет, полученный в п. 5 в пространства HSB, HSI
# 5. Осуществить обратное преобразование в LAB из всех полученных 
# пространств в п. 2, 3, 4
# 6. Рассчитать ∆E, ∆E94, ∆E00 между цветом, заданным в п.1 и цветами 
# полученными в п.5
# 7. Оценить цветовые различи

#Цвет 50.6, -1.9, -53.26 
lab = [50.6, -1.9, -53.26]
import math
# LAB -> LCH
lch = [lab[0], (lab[1]**2 + lab[2]**2)**(1/2), -math.atan(lab[2]/lab[1]) * 360/math.tau]
if lch[2]<0: lch[2] += 360
print(f'LCH = {lch}')
#LCH = [50.6, 53.29387957354953, -87.95689381007803]

# LCH -> RGB
import colormath.color_objects, colormath.color_conversions, colormath.color_diff
lch = colormath.color_objects.LCHabColor(*lch)
srgb = colormath.color_conversions.convert_color(lch, colormath.color_objects.sRGBColor)
print('sRGB =',srgb.rgb_r*256, srgb.rgb_g*256, srgb.rgb_b*256, )
#sRGB= 54.27626512216557 123.1278214982287 212.82692597990544

# RGB -> HSB
r = srgb.rgb_r
g = srgb.rgb_g
b = srgb.rgb_b
rgbmax = max(r, g, b)
rgbmin = min(r, g, b)
if rgbmax == rgbmin: h=0
elif rgbmax==r: h = 60*((g-b)/(rgbmax - rgbmin))
elif rgbmax==g: h = 60*(2 + (b-r)/(rgbmax - rgbmin))
elif rgbmax==b: h = 60*(4 + (r-g)/(rgbmax - rgbmin))
if h<0: h+=360
if rgbmax == 0: s=0
else: s= (rgbmax-rgbmin)/rgbmax
b=rgbmax
hsb = [h, s*100, b*255]
print('HSB =', h, s, b)
#HSB = 213.94464734353625 0.7449746319819983 0.8313551796090056

# RGB -> HSI
r = srgb.rgb_r
g = srgb.rgb_g
b = srgb.rgb_b
i=(r+g+b)/3
r=r/(r+g+b)
g=g/(r+g+b)
b=b/(r+g+b)
h= math.acos((0.5 * ((r - g) + (r - b))) / (math.sqrt((r - g) * (r - g) + (r - b) * (g - b))))
if b>g:
    h=2*math.pi - h
s= 1-3*min(r, g, b)
hsi = [h*180/math.pi, s*100, i*255]
print('HSI =', h*180/math.pi, s, i)
#HSI = 217.56714479531732 0.5827374296023052 0.50811329765664

# LCH -> LAB
lab = colormath.color_conversions.convert_color(lch, colormath.color_objects.LabColor)
print('LCH -> LAB =',lab.lab_l, lab.lab_a, lab.lab_b)

# sRGB -> LAB
lab = colormath.color_conversions.convert_color(srgb, colormath.color_objects.LabColor)
print('sRGB -> LAB =',lab.lab_l, lab.lab_a, lab.lab_b)

# HSB ->LAB
hsb = colormath.color_objects.HSVColor(hsb[0],hsb[1]/100,hsb[2]/255)
lab = colormath.color_conversions.convert_color(hsb, colormath.color_objects.LabColor)
print('HSB -> LAB =',lab.lab_l, lab.lab_a, lab.lab_b)

# HSI -> LAB
h, s, i = hsi
s/=100
i/=255
print(h, s, i)
hx = h/60
z = 1 - abs(hx%2-1)
c=(3*i*s)/(1+z)
x=c*z
if hx==0: r, g, b = 0,0,0
elif 0<=hx<=1: r,g,b=c,x,0
elif 1<=hx<=2: r,g,b=x,c,0
elif 2<=hx<=3: r,g,b=0,c,x
elif 3<=hx<=4: r,g,b=0,x,c
elif 4<=hx<=5: r,g,b=x,0,c
elif 5<=hx<=6: r,g,b=c,0,x
m = i*(1-s)
r,g,b = r+m, g+m, b+m
print('HSI -> RGB =',r*255,g*255,b*255)
#HSI -> RGB = 54.06424846153194 115.70658901685765 218.93583522894005
rgb = colormath.color_objects.sRGBColor(r,g,b)
lab = colormath.color_conversions.convert_color(rgb, colormath.color_objects.LabColor)
print('HSI -> LAB =',lab.lab_l, lab.lab_a, lab.lab_b)
#HSI -> LAB = 49.88056754072717 15.940195858011485 -58.552506242847755

def e(l1, a1, b1 ,l2, a2,b2):
    return ((l1-l2)**2 + (a1-a2)**2 + (b1-b2)**2)**(1/2)

def e94(l1, a1, b1 ,l2, a2,b2):
    dl = l1-l2
    c1=math.sqrt(a1**2 + b1**2)
    c2=math.sqrt(a2**2 + b2**2)
    dc = c1-c2
    dh = math.sqrt((a2-a1)**2 + (b2-b1)**2 - dc**2)
    sl=1
    kl=1
    k1=0.045
    k2=0.015
    kc=1
    kh=1
    sc=1+k1*c1
    sh = 1+k2*c1
    color1= colormath.color_objects.LabColor(l1, a1, b1)
    color2= colormath.color_objects.LabColor(l2, a2, b2)
    return math.sqrt((dl/(kl*sl))**2 + (dc/(kc*sc))**2 + (dh/(kh*sh))**2), colormath.color_diff.delta_e_cie1994(color1, color2)

def e2000(l1, a1, b1 ,l2, a2,b2):
    color1= colormath.color_objects.LabColor(l1, a1, b1)
    color2= colormath.color_objects.LabColor(l2, a2, b2)
    return colormath.color_diff.delta_e_cie2000(color1, color2)

print('LAB - LCH - LAB')
print(e(50.6, 1.9, -53.3, 50.6, 1.9, -53.3,))
print(e94(50.6, 1.9, -53.3, 50.6, 1.9, -53.3,))
print(e2000(50.6, 1.9, -53.3, 50.6, 1.9, -53.3,))

print('LAB - RGB - LAB')
print(e(50.6, 1.9, -53.3, 51.4, 9.1, -52.2,))
print(e94(50.6, 1.9, -53.3, 51.4, 9.1, -52.2,))
print(e2000(50.6, 1.9, -53.3, 51.4, 9.1, -52.2,))

print('LAB - HSB - LAB')
print(e(50.6, 1.9, -53.3, 51.4, 9.1, -52.2,))
print(e94(50.6, 1.9, -53.3, 51.4, 9.1, -52.2,))
print(e2000(50.6, 1.9, -53.3, 51.4, 9.1, -52.2,))

print('LAB - HSI - LAB')
print(e(50.6, 1.9, -53.3, 49.9, 15.9, -58.6,))
print(e94(50.6, 1.9, -53.3, 49.9, 15.9, -58.6,))
print(e2000(50.6, 1.9, -53.3, 49.9, 15.9, -58.6,))