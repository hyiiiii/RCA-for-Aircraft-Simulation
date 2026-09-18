"""Consistent checkmark indicators, independent of the platform's ttk theme."""
import tkinter as tk
from tkinter import ttk


def install_checkmarks(widget):
    root=widget._root()
    if getattr(root,'_checkmark_images',None):return
    images=[]
    for selected,disabled in [(False,False),(True,False),(False,True),(True,True)]:
        image=tk.PhotoImage(master=root,width=20,height=20)
        border='#aeb8c1' if disabled else '#647b8e'
        fill='#d4dbe1' if disabled and selected else '#247caa' if selected else '#ffffff'
        image.put(border,to=(1,1,19,19));image.put(fill,to=(2,2,18,18))
        if selected:
            # Thick, unmistakable check, never the clam theme's cross.
            for a,b in [((5,10),(8,13)),((8,13),(15,6))]:
                steps=max(abs(b[0]-a[0]),abs(b[1]-a[1]))
                for i in range(steps+1):
                    x=round(a[0]+(b[0]-a[0])*i/steps);y=round(a[1]+(b[1]-a[1])*i/steps)
                    image.put('#ffffff',to=(x-1,y-1,x+2,y+2))
        images.append(image)
    root._checkmark_images=images
    style=ttk.Style(root)
    style.element_create('Checked.indicator','image',images[0],('disabled','selected',images[3]),('disabled',images[2]),('selected',images[1]),width=24,sticky='w')
    style.layout('TCheckbutton',[('Checkbutton.padding',{'sticky':'nswe','children':[
        ('Checked.indicator',{'side':'left','sticky':'w'}),
        ('Checkbutton.focus',{'side':'left','sticky':'w','children':[('Checkbutton.label',{'sticky':'nswe'})]})]})])
