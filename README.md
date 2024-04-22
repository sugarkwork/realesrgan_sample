# realesrgan_sample

realesrgan をライブラリとして使用したサンプルコードです。

画像を4倍に拡大します。

マルチプロセッシングを使用して、同時に複数のインスタンスを立ち上げて、複数のグラボを同時に使う事が出来ます。

GPU が使用できない時は CPU が使われますが、めちゃめちゃ遅いです。

torch と torchvision は GPU 対応しているやつをインストールすると、GPU が有効になります。

例：

    pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# code

コードを見ると分かりますが、第1引数は GPU の ID です。GPU メモリを圧迫するだけなので、VRAM に余裕があれば複数インスタンス立ち上げられます。

    upscaler_wrapper1 = ImageUpscaler(0)
    upscaler_wrapper2 = ImageUpscaler(1)
    
    upscaler_wrapper1.upscale('input1.jpg', 'output1.png')
    upscaler_wrapper2.upscale('input2.png', 'output2.png')
    
    upscaler_wrapper1.stop()
    upscaler_wrapper2.stop()

upscale メソッドは非同期なキューを入れてるだけなので、いつ終わるか分かりません。
気になる人は自分で実装してください。
