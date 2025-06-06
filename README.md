# 簡介
這是結合openCV與機器學習的猜拳遊戲，透過鏡頭辨識玩家手勢，並使用隨機森林模型進行判斷與對戰。系統包含訓練與對戰兩階段，支援三種難度，提供即時互動與AI對戰體驗。
# 需下載的東東
* 下載
```
!pip install opencv-python # 下載opencv
!pip install numpy # 下載numpy
!pip install scikit-learn # 下載sklearn
```
* IDE(Win跟Mac) 版：
    * 打開IDE，找到terminal把下面的東西塞進去
    ![螢幕擷取畫面 2025-04-30 004410](https://hackmd.io/_uploads/HJcfMYC1ee.png)
* Win 10/11 版：
    * win + R 打開執行，輸入cmd或powershell(都一樣啦)，一樣把下面的東西塞進去
    ![螢幕擷取畫面 2025-04-30 004525](https://hackmd.io/_uploads/SyYEGK0Jxe.png) 
    或
    ![螢幕擷取畫面 2025-04-30 004556](https://hackmd.io/_uploads/S1yLfK01ee.png)
    打開長這樣
    ![螢幕擷取畫面 2025-04-30 004628](https://hackmd.io/_uploads/BJGnzY0ygx.png)

* Mac 版：
    * Command + space，繼續把下面的東西塞進去
   *~~窩與不知道，沒錢買Macbook~~*
# 往後可優化的東西
沒有用DataBase，問就是怕爆
所以現在是把東西全部塞在Ram裡面，所以一定要釋放資源

# IDE

我只有用Vscode跟Pycharm，其他的窩不知道，但應該不會有問題(除非你用競程專用IDE E.g.CP editor)
![窩不知道](https://hackmd.io/_uploads/r1b5PKAkgx.jpg)

# 版本
查版本，cmd key
```
pip show 你想查的模組
```

我是用
Python **3.12.9** (不是Python2

OpenCV **4.11.0.86**

NumPY **1.26.4**

Sklearn **1.6.1**

######## 不要北七去亂下載其他version，應該不會有問題 ########
# 硬體
因為CPU的頻率和暫存記憶體會有差，所以我提供我的硬體，~~想炫耀~~
> CPU - Intel® Core™ i5-14500
> 
> MB - PRO B760M-A DDR4 II
> 
> RAM - DDR4 3200Hz 32GB
> 
> GPU - INNO3D GeForce RTX™ 4070 TWIN X2 OC WHITE


