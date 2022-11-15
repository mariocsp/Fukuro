import cv2 as cv
import numpy as np

frame = cv.imread('src/fukuro/scripts/tes1.png')
cv.namedWindow('image')
cv.namedWindow('Asli',cv.WINDOW_NORMAL)
cv.namedWindow('thresh',cv.WINDOW_NORMAL)

def callback(x):
    pass

cv.createTrackbar('lowB','image',0,255,callback)
cv.createTrackbar('highB','image',17,255,callback)

cv.createTrackbar('lowG','image',0,255,callback)
cv.createTrackbar('highG','image',131,255,callback)

cv.createTrackbar('lowR','image',0,255,callback)
cv.createTrackbar('highR','image',255,255,callback)

cv.createTrackbar('eksizeO','image',0,100,callback)
cv.createTrackbar('dksizeO','image',0,100,callback)

cv.createTrackbar('eksizeC','image',0,100,callback)
cv.createTrackbar('dksizeC','image',0,100,callback)


while True:
    
    
    low_b = cv.getTrackbarPos('lowB', 'image')
    high_b = cv.getTrackbarPos('highB', 'image')
    low_g = cv.getTrackbarPos('lowG', 'image')
    high_g = cv.getTrackbarPos('highG', 'image')
    low_r = cv.getTrackbarPos('lowR', 'image')
    high_r = cv.getTrackbarPos('highR', 'image')
    eksizesO = cv.getTrackbarPos("eksizeO",'image')
    dksizesO = cv.getTrackbarPos("dksizeO",'image')
    eksizesC = cv.getTrackbarPos("eksizeC",'image')
    dksizesC = cv.getTrackbarPos("dksizeC",'image')

    low  = (low_b,low_g,low_r)
    high = (high_b,high_g,high_r)
    thresh = cv.inRange(frame,low,high)

    elemenEO = cv.getStructuringElement(cv.MORPH_RECT,(eksizesO*2+1,eksizesO*2+1))
    elemenDO = cv.getStructuringElement(cv.MORPH_RECT,(dksizesO*2+1,dksizesO*2+1))
    
    elemenEC = cv.getStructuringElement(cv.MORPH_RECT,(eksizesC*2+1,eksizesC*2+1))
    elemenDC = cv.getStructuringElement(cv.MORPH_RECT,(dksizesC*2+1,dksizesC*2+1))
    
    
    thresh = cv.erode(thresh,elemenEO)
    thresh = cv.dilate(thresh,elemenDO) 

    thresh = cv.dilate(thresh,elemenDC)
    thresh = cv.dilate(thresh,elemenEC)

    
    
    cv.imshow('Asli',frame)
    cv.imshow('thresh',thresh)

    if cv.waitKey(1) == ord("q"):
        print(
            f"""
        Batas low = {low}
        Batas high = {high}
        Kernel size EO = {eksizesO}
        Kernel size DO = {dksizesO}
        Kernel size DC = {dksizesC}
        Kernel size EC = {eksizesC}
        """
        )
        break

kontur, hierarchy = cv.findContours(thresh,cv.RETR_TREE,cv.CHAIN_APPROX_NONE)

for i in kontur:
    epsilon = 0.1*cv.arcLength(i,True)
    approx = cv.approxPolyDP(i,epsilon,True)
    print(approx.shape)

print(len(kontur))
cv.imwrite("hasil.jpg",thresh)
cv.destroyAllWindows()
    