def get_embedding_img(agent, state):
    img = agent.embed.predict_on_batch(np.expand_dims(state, axis=0))
    return img

def save_debug_img(pre_train, post_train, global_step):
    print(f"pretrain: max -> {np.max(pre_train)} min -> {np.min(pre_train)}")
    print(f"posttrain: max -> {np.max(post_train)} min -> {np.min(post_train)}")
    plot, axes = plt.subplots(ncols=2)
    axes[0].imshow(pre_train.numpy().squeeze().reshape((28, 28)))
    axes[1].imshow(post_train.numpy().squeeze().reshape((28, 28)))
    path = f"/home/julius/results/img/check_iv_step_{global_step}"
    plt.savefig(path)
    plt.close()