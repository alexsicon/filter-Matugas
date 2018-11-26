from imutils.video import VideoStream
import imutils
import cv2

faceCascade = cv2.CascadeClassifier("./data/haarcascade_frontalface_default.xml")
noseCascade = cv2.CascadeClassifier("./data/haarcascade_mcs_nose.xml")

def clipping(x1,x2,y1,y2,w,h):
    if x1 < 0:
        x1 = 0
    if y1 < 0:
        y1 = 0
    if x2 > w:
        x2 = w
    if y2 > h:
        y2 = h
    x1 = int(x1)
    x2 = int(x2)
    y1 = int(y1)
    y2 = int(y2)
    return x1,x2,y1,y2

def adjust(a,W,H,org,inv):
    filtercall = cv2.resize(a,(W,H),interpolation=cv2.INTER_AREA)
    mask = cv2.resize(org,(W,H),interpolation=cv2.INTER_AREA)
    mask_inv = cv2.resize(inv,(W,H),interpolation=cv2.INTER_AREA)
    return filtercall,mask,mask_inv

def roi(x1,x2,y1,y2,filtercall,mask,mask_inv):
    roi = roic[y1:y2, x1:x2]
    roibg = cv2.bitwise_and(roi,roi,mask=mask_inv)
    roifg = cv2.bitwise_and(filtercall,filtercall,mask=mask)
    dst = cv2.add(roibg, roifg)
    roic[y1:y2, x1:x2] = mu

def mustache(nx,ny,nw,nh,w,h):
    must = cv2.imread('./filters/mustache.png',-1)
    must_orgmask = must[:,:,3]
    must_invmask = cv2.bitwise_not(must_orgmask)
    must = must[:,:,0:3]
    mustOH, mustOW = must.shape[:2]

    mustW = 2 * nw
    mustH = mustW * mustOH / mustOW
    x1 = nx - (mustW/4)
    x2 = nx + nw + (mustW/4)
    y1 = ny + nh - (mustH/2)
    y2 = ny + nh + (mustH/2)
    delta = clipping(x1,x2,y1,y2,w,h)
    mustW = (int(delta[1]) - int(delta[0]))
    mustH = (int(delta[3]) - int(delta[2]))
    kappa = adjust(must,mustW,mustH,must_orgmask,must_invmask)
    roi = roic[delta[2]:delta[3], delta[0]:delta[1]]
    roibg = cv2.bitwise_and(roi,roi,mask=kappa[2])
    roifg = cv2.bitwise_and(kappa[0],kappa[0],mask=kappa[1])
    dst = cv2.add(roibg, roifg)
    roic[delta[2]:delta[3], delta[0]:delta[1]] = dst

def sunglasses(nx,ny,nw,nh,w,h):
    sg = cv2.imread('./filters/sunglasses2.png',-1)
    sg_orgmask = sg[:,:,3]
    sg_invmask = cv2.bitwise_not(sg_orgmask)
    sg = sg[:,:,0:3]
    sgOH, sgOW = sg.shape[:2]

    sgW = 5 * nw
    sgH = sgW * sgOH / sgOW
    x1 = nx - (sgW/4)
    x2 = nx + nw + (sgW/4)
    y1 = 0.4 * ny
    y2 = 1.3 * ny
    delta = clipping(x1,x2,y1,y2,w,h)
    sgW = (int(delta[1]) - int(delta[0]))
    sgH = (int(delta[3]) - int(delta[2]))
    kappa = adjust(sg,sgW,sgH,sg_orgmask,sg_invmask)
    roi = roic[delta[2]:delta[3], delta[0]:delta[1]]
    roibg = cv2.bitwise_and(roi,roi,mask=kappa[2])
    roifg = cv2.bitwise_and(kappa[0],kappa[0],mask=kappa[1])
    dst = cv2.add(roibg, roifg)
    roic[delta[2]:delta[3], delta[0]:delta[1]] = dst

def flower(nx,ny,nw,nh,w,h):
    f = cv2.imread('./filters/flower.png',-1)
    f_orgmask = f[:,:,3]
    f_invmask = cv2.bitwise_not(f_orgmask)
    f = f[:,:,0:3]
    fOH, fOW = f.shape[:2]

    fW = 10 * nw
    fH = fW * fOH / fOW
    x1 = nx - (fW/4)
    x2 = nx + nw + (fW/4)
    y1 = 0
    y2 = 0.25 * ny + nh
    delta = clipping(x1,x2,y1,y2,w,h)
    fW = (int(delta[1]) - int(delta[0]))
    fH = (int(delta[3]) - int(delta[2]))
    kappa = adjust(f,fW,fH,f_orgmask,f_invmask)
    roi = roic[delta[2]:delta[3], delta[0]:delta[1]]
    roibg = cv2.bitwise_and(roi,roi,mask=kappa[2])
    roifg = cv2.bitwise_and(kappa[0],kappa[0],mask=kappa[1])
    dst = cv2.add(roibg, roifg)
    roic[delta[2]:delta[3], delta[0]:delta[1]] = dst

vs = VideoStream(src=0).start()
while True:
    frame = vs.read()
    frame = imutils.resize(frame,width=470)
    gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    faces = faceCascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30,30), flags=cv2.CASCADE_SCALE_IMAGE)
    for (x,y,w,h) in faces:
        roig = gray[y:y+h, x:x+w]
        roic = frame[y:y+h, x:x+w]
        nose = noseCascade.detectMultiScale(roig)
        for (nx,ny,nw,nh) in nose:
            mustache(nx,ny,nw,nh,w,h)
            sunglasses(nx,ny,nw,nh,w,h)
            flower(nx,ny,nw,nh,w,h)
    cv2.imshow("Filters",frame)
    key = cv2.waitKey(1)
    if key == ord("q"):
        break

cv2.destroyAllWindows()
vs.stop()
