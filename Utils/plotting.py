import matplotlib.pyplot as plt 


def plot_image_grid(imgs):# TODO: check type
    if imgs.ndim!=4 or imgs.shape[:2]!=(8,8):
        raise ValueError('Imgs Dim Error!')
    
    fig=plt.figure(figsize=(12,12),constrained_layout=True)
    gs=fig.add_gridspec(8,8)

    for n_row in range(8):
        for n_col in range(8):
            f_ax=fig.add_subplot(gs[n_row,n_col])
            image_data=(imgs[n_row,n_col]+1.)*255/2
            f_ax.imshow(image_data,cmap='gray')
            f_ax.axis('off')
    return  fig