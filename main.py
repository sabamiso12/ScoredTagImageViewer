
import time
import re
import os
import glob
import json
import tkinter as tk
from PIL import ImageTk, Image
import numpy as np
import pyperclip

# スペース改行を無視してコンマ区切りをマッチ
re_separater = re.compile("[\t\n\r ]*,[\t\n\r ]*")

# 整数にマッチ
re_int = re.compile('[0-9]+')

# 各種メタタグにマッチ
re_removed = re.compile("REMOVED.*")
re_remove_duplicated = re.compile("REMOVED_DUP")
re_duplicated = re.compile("DUP_GROUP.*")
re_dups = re.compile("REMOVED_DUP|DUP_GROUP.*|SIMILARITY.*")


def valid_isnum(S):
    # 入力制限用関数
    if re_int.match(S):
        return True
    return False


# デバッグ用print
test = False


def tprint(arg, **args):
    if test:
        print(arg, **args)


def str2int(num):
    # 文字列をintにする。空はゼロ、エラーがあったらNone
    if num == "":
        return 0
    try:
        return int(num)
    except ValueError:
        return None


def str2npf(num):
    # 文字列をfloat32にする。空はゼロ、エラーがあったらNone
    if num == "":
        return np.float32(0)
    try:
        return np.float32(num)
    except ValueError:
        return None


def toggle_object(widget: tk.Widget, side=tk.TOP):
    # widgetを出したり消したりする
    try:
        tprint(widget.pack_info())
        widget.pack_forget()
    except tk.TclError:
        widget.pack(side=side)


def keep_resize(img: Image.Image, size: tuple, outfit_degree: float, resample):
    # 比率保ってリサイズ
    # outfit_degreeが0でinnerfit=内側をpadding、1でoutfit=外側をcrop、中間も可
    org_w, org_h = img.width, img.height
    w, h = size
    min_ratio = min(w/org_w, h/org_h)
    max_ratio = max(w/org_w, h/org_h)
    ratio = outfit_degree * max_ratio + (1-outfit_degree)*min_ratio
    r_w, r_h = int(ratio*org_w), int(ratio*org_h)
    img = img.resize((r_w, r_h), resample=resample)
    img = img.crop((r_w//2 - w//2, r_h//2 - h//2, r_w//2 + w//2 + w % 2, r_h//2 + h//2 + w % 2))
    return img


def read_json_file(file_path):
    # scoredTagのjsonを読み込み
    with open(file_path, "r") as f:
        json_data = json.load(f)
    # wd-14-taggerでは[0]がrating,[1]がtag
    return json_data[0] | json_data[1]


class FoldingLabel(tk.Label):
    def __init__(self, master, target, text, **args) -> None:
        super().__init__(master, text=text ** args)
        self.label.bind("<Button-1>", lambda event: toggle_object(target))


class DataManager:
    def __init__(self):
        self.all_images = []
        self.selected_image = None
        self.images_dir = ""
        self.tagtexts_dir = ""
        self.jsons_dir = ""

    def load_data(self, images_dir, tagtexts_dir, jsons_dir):
        if tagtexts_dir == "":
            tagtexts_dir = images_dir
        if jsons_dir == "":
            jsons_dir = images_dir
        self.all_images = []
        self.selected_image = None
        self.jsons_dir = os.path.normpath(jsons_dir)
        self.images_dir = os.path.normpath(images_dir)
        self.tagtexts_dir = os.path.normpath(tagtexts_dir)
        print(f"Loading images from {images_dir}")
        print(f"Loading tags from {tagtexts_dir}")
        print(f"Loading score jsons from {jsons_dir}")
        image_ext = ["png", "jpeg", "jpg"]
        tagtext_ext = "txt"
        jsons = glob.glob(jsons_dir + "/*.json")
        print(len(jsons))
        # jsonと画像両方あるやつだけ読み込む
        for json_path in jsons:
            name = os.path.splitext(os.path.basename(json_path))[0]
            # 画像があるか探す
            image_path = None
            for ext in image_ext:
                tmp_path = os.path.join(images_dir, name + "." + ext)
                if os.path.exists(tmp_path):
                    image_path = tmp_path
                    break
            if image_path is None:
                print(f"Error:Image {name} not found.")
                continue
            # tagを読み込む
            tagtext_path = os.path.join(tagtexts_dir, name + "." + tagtext_ext)
            if os.path.exists(tagtext_path):
                with open(tagtext_path, "r") as f:
                    tags = [t.strip() for t in f.read().split(",")]
            else:
                tags = []
            scores = read_json_file(json_path)
            if TaggedImage.all_tags == []:
                # タグのリストを作っておく。順番大事
                TaggedImage.all_tags = list(scores.keys())
                # フィルタ/ソート用にタグ順のindexとアルファベット順のindexを作っておく
                TaggedImage.all_indexes = np.arange(len(TaggedImage.all_tags))
                sorted_tags = sorted(TaggedImage.all_tags)
                TaggedImage.all_alph_order = np.array([sorted_tags.index(tag) for tag in TaggedImage.all_tags])
            scores = np.array([float(scores[t]) for t in TaggedImage.all_tags], dtype=np.float32)
            self.all_images.append(TaggedImage(name=name, image_path=image_path, img=None,
                                   json_path=json_path, scores=scores, tagtext_path=tagtext_path, tags=tags))
        print(f"{len(self.all_images)} images are loaded.")

    def set_selected_image(self, image):
        self.selected_image = image


class TaggedImage:
    thumbnail_size = (100, 100)
    all_tags = []
    all_indexes = []
    all_alph_order = []

    def __init__(self, name, image_path, img, json_path, scores, tagtext_path, tags):
        self.name = name
        self.image_path = image_path
        self.img = img
        self.thumbnail = img
        self.json_path = json_path
        self.scores = scores
        # self.scores_normal = scores/np.linalg.norm(scores)

        self.tagtext_path = tagtext_path
        self.fixedtags = tags
        self.metatags = []

    def search_fixedtags(self, tag_exp):
        return [tag for tag in self.fixedtags if re.match(tag_exp, tag)]

    def search_metatags(self, tag_exp):
        return [tag for tag in self.metatags if re.match(tag_exp, tag)]

    def subst_fixedtags(self,  match_item, sub_to):
        self.fixedtags = [match_item.sub(tag, sub_to) for tag in self.fixedtags]

    def set_metatags(self, tags: list):
        if isinstance(tags, str):
            tags = [tags]
        self.metatags = tags

    def set_fixedtags(self, tags: list):
        if isinstance(tags, str):
            tags = [tags]
        self.fixedtags = tags

    def add_metatags(self, tags, prepend=False, uniq=True):
        if isinstance(tags, str):
            tags = [tags]
        if uniq:
            tags = [tag for tag in tags if tag != "" and tag not in self.metatags]
        if prepend:
            self.metatags = tags + self.metatags
        else:
            self.metatags += tags

    def add_fixedtags(self, tags, prepend=False, uniq=True):
        if isinstance(tags, str):
            tags = [tags]
        if uniq:
            tags = [tag for tag in tags if tag != "" and tag not in self.fixedtags]
        if prepend:
            self.fixedtags = tags + self.fixedtags
        else:
            self.fixedtags += tags

    def add_fixedtags_by_index(self, tag_indexes, prepend=False, uniq=True):
        tags = [__class__.all_tags[i] for i in tag_indexes]
        if uniq:
            tags = [tag for tag in tags if tag != "" and tag not in self.fixedtags]
        if prepend:
            self.fixedtags = tags + self.fixedtags
        else:
            self.fixedtags += tags

    def uniquefy_fixedtags(self):
        self.fixedtags = sorted(set(self.fixedtags) - {""}, key=self.fixedtags.index)

    def remove_search_metatags(self, tag_exp):
        self.metatags = [tag for tag in self.metatags if not re.match(tag_exp, tag)]

    def remove_metatags(self, tags):
        if isinstance(tags, str):
            tags = [tags]
        self.metatags = [tag for tag in self.metatags if tag not in tags]

    def remove_fixedtags(self, tags):
        if isinstance(tags, str):
            tags = [tags]
        self.fixedtags = [tag for tag in self.fixedtags if tag not in tags]

    def remove_fixedtags_by_index(self, tag_indexes):
        tags = [__class__.all_tags[i] for i in tag_indexes]
        self.fixedtags = [tag for tag in self.fixedtags if tag not in tags]

    def get_img(self):
        cache_img = False
        # キャッシュがあればそれを返す
        if self.img is not None:
            return self.img
        # なければ取得
        img = Image.open(self.image_path)
        # キャッシュする
        if cache_img:
            self.img = img
        return img

    def get_thumbnail(self):
        cache_thumbnail = True
        # キャッシュがあればそれを返す
        if self.thumbnail is not None:
            return self.thumbnail
        # なければ取得
        thumbnail = keep_resize(self.get_img(), __class__.thumbnail_size, 0.5, Image.Resampling.LANCZOS)
        # キャッシュする
        if cache_thumbnail:
            self.thumbnail = thumbnail
        return thumbnail

    def get_score(self, tag):
        return self.scores[__class__.all_tags.index(tag)]


class ImageFrame(tk.Frame):
    def on_event(x, y): return x

    def __init__(self, master, imsize, index, **args) -> None:
        self.highlight = False
        self.index = index
        self.frame_img = None
        self.width, self.height = imsize
        super().__init__(master, width=self.width+2, height=self.height+2, **args)

    def set_image(self, image):
        self.clear_image()
        if image is not None:
            photo = ImageTk.PhotoImage(image.get_thumbnail())
            self.frame_img = tk.Label(self, image=photo)
            self.bind_keys()
            self.frame_img.image = photo
            self.frame_img.pack(padx=1, pady=1)
            self.update_frame()

    def update_frame(self):
        if self.highlight:
            self.configure(background="blue")
        else:
            self.configure(background="white")
        self.update()

    def set_highlight(self, highlight):
        if highlight is None:
            self.highlight = not self.highlight
        else:
            self.highlight = highlight
        self.update_frame()

    def bind_keys(self):
        # self.bind("<Button-1>", lambda event: __class__.on_event(event, self.index))
        self.bind("<MouseWheel>", lambda event: __class__.on_event(event, self.index))
        if self.frame_img is not None:
            self.frame_img.bind("<Button-1>", lambda event: __class__.on_event(event, self.index))
            self.frame_img.bind("<MouseWheel>", lambda event: __class__.on_event(event, self.index))

    def clear_image(self):
        # self.image = None
        if self.frame_img is not None:
            self.frame_img.destroy()
            self.frame_img = None


class GalleryFrame(tk.Frame):
    def __init__(self, master, row=7, column=6, **args) -> None:
        self.selected = None
        self.column = column
        self.num_view = column * row
        pad = 1
        super().__init__(master, **args)
        imsize = TaggedImage.thumbnail_size
        # 画像を表示
        self.image_frames = [ImageFrame(self, imsize=imsize, index=i) for i in range(column*row)]
        for i, frame in enumerate(self.image_frames):
            frame.propagate(False)
            frame.grid(row=i//column, column=i % column)

    def set_images(self, images):
        # 各フレームに画像を設定
        images = images[0:self.num_view]
        for i, frame in enumerate(self.image_frames):
            if i < len(images):
                frame.set_image(images[i])
            else:
                frame.set_image(None)
            self.update()

    def set_select(self, index):
        # 選択されたindexを保存し、ハイライト判定
        if self.selected is not None:
            self.image_frames[self.selected].set_highlight(False)
            self.image_frames[self.selected].update_frame()
        if index is not None:
            self.image_frames[index].set_highlight(True)
        self.selected = index

    def bind_on_images(self, fn):
        # クラスアトリビュート経由でfnをバインドする
        ImageFrame.on_event = fn
        for frame in self.image_frames:
            frame.bind_keys()


class FilteredGalleryFrame(tk.Frame):
    def __init__(self, master, data, **args) -> None:
        super().__init__(master, **args)
        self.page = 1
        self.data = data
        self.filtered_images0 = data.all_images
        self.filtered_images1 = self.filtered_images0
        self.filtered_images2 = self.filtered_images1
        self.filtered_images3 = self.filtered_images2
        self.filtered_images = self.filtered_images3
        self.filter1_exp = ""
        self.filter2_exp = ""
        self.filter3_exp = ""
        self.sort_text = ""
        self.gallery_frame = GalleryFrame(self)
        self.filter_frame = ImageFilterUIFrame(self)
        # filterのボタン設定。テキストボックスにエンターしたときも
        self.filter_frame.sort_txt.bind('<Return>', lambda event: self.refresh_images(sort_text=self.filter_frame.sort_txt.get()))
        self.filter_frame.sort_button.config(command=lambda: self.refresh_images(sort_text=self.filter_frame.sort_txt.get()))
        self.filter_frame.filter1_txt.bind('<Return>', lambda event: self.refresh_images(filter1_exp=self.filter_frame.filter1_txt.get()))
        self.filter_frame.filter1_button.config(command=lambda: self.refresh_images(filter1_exp=self.filter_frame.filter1_txt.get()))
        self.filter_frame.filter1_neg_cbt.config(command=lambda: self.refresh_images(filter1_exp=self.filter_frame.filter1_txt.get()))
        self.filter_frame.filter2_txt.bind('<Return>', lambda event: self.refresh_images(filter2_exp=self.filter_frame.filter2_txt.get()))
        self.filter_frame.filter2_button.config(command=lambda: self.refresh_images(filter2_exp=self.filter_frame.filter2_txt.get()))
        self.filter_frame.filter2_neg_cbt.config(command=lambda: self.refresh_images(filter2_exp=self.filter_frame.filter2_txt.get()))
        self.filter_frame.filter3_txt.bind('<Return>', lambda event: self.refresh_images(filter3_exp=self.filter_frame.filter3_txt.get()))
        self.filter_frame.filter3_button.config(command=lambda: self.refresh_images(filter3_exp=self.filter_frame.filter3_txt.get()))
        self.filter_frame.filter3_neg_cbt.config(command=lambda: self.refresh_images(filter3_exp=self.filter_frame.filter3_txt.get()))
        self.filter_frame.detect_dup_button.config(command=lambda: self.detect_duplicated(thresh=str2npf(self.filter_frame.detect_dup_thresh.get())))
        self.filter_frame.option_apply_btn.config(command=lambda: self.refresh_images(all=True))
        self.filter_frame.prev_button.config(command=lambda: self.change_page(self.page-1))
        self.filter_frame.next_button.config(command=lambda: self.change_page(self.page+1))
        self.filter_frame.page_text.bind('<Return>', lambda event: self.change_page(self.filter_frame.get_page_num()))
        self.filter_frame.order_ascend_button.config(command=lambda: self.refresh_images(sort_text=self.filter_frame.sort_txt.get()))
        self.filter_frame.order_descend_button.config(command=lambda: self.refresh_images(sort_text=self.filter_frame.sort_txt.get()))
        self.gallery_frame.pack()
        self.filter_frame.pack()
        self.refresh_images()

    def update_gallery(self) -> None:
        self.filter_frame.set_page_num(self.page)
        self.gallery_frame.set_select(None)
        n = self.gallery_frame.num_view
        if self.page == self.max_page and False:
            self.gallery_frame.set_images(images[-(n):])
        else:
            self.gallery_frame.set_images(self.filtered_images[(self.page - 1) * n:self.page*n])
        return self.update()

    def refresh_images(self, filter1_exp=None, filter2_exp=None, filter3_exp=None, sort_text=None, skip_filter=None, all=False, keep_pos=False):
        if all:
            self.filtered_images0 = self.meta_tags_filter(self.data.all_images)
            self.filtered_images1 = self.filtered_images0
            self.filtered_images2 = self.filtered_images1
            self.filtered_images3 = self.filtered_images2
            self.filtered_images = self.filtered_images3
            filter1_exp = self.filter1_exp
            filter2_exp = self.filter2_exp
            filter3_exp = self.filter3_exp
            sort_text = self.sort_text
        # ソートだけする場合はフィルター処理スキップ
        if skip_filter:
            images = self.filtered_images
            images = self.sort_images(images, sort_text)
            if sort_text is not None:
                self.sort_text = sort_text
            self.change_page(1)
            return
        # 上流が変更されたら下流のフィルタも再適用
        if filter1_exp is not None:
            if filter2_exp is None:
                filter2_exp = self.filter2_exp
        if filter2_exp is not None:
            if filter3_exp is None:
                filter3_exp = self.filter3_exp
        # sortは毎回適用
        if sort_text is None:
            sort_text = self.sort_text
        images = self.filter_images(self.filtered_images0, filter1_exp, negative=self.filter_frame.filter1_neg.get())
        # フィルターの結果Noneだったら(エラー時)何もしない
        if images is None:
            return None
        if filter1_exp is not None:
            self.filtered_images1 = images
        images = self.filter_images(self.filtered_images1, filter2_exp, negative=self.filter_frame.filter2_neg.get())
        if images is None:
            return None
        if filter2_exp is not None:
            self.filtered_images2 = images
        images = self.filter_images(self.filtered_images2, filter3_exp, negative=self.filter_frame.filter3_neg.get())
        if images is None:
            return None
        if filter3_exp is not None:
            self.filtered_images3 = images
        images = self.sort_images(self.filtered_images3, sort_text)
        if images is None:
            return None
        self.filtered_images = images
        if filter1_exp is not None:
            self.filter1_exp = filter1_exp
        if filter2_exp is not None:
            self.filter2_exp = filter2_exp
        if filter3_exp is not None:
            self.filter3_exp = filter3_exp
        if sort_text is not None:
            self.sort_text = sort_text
        self.max_page = (len(images) - 1) // self.gallery_frame.num_view + 1
        self.filter_frame.maxpage_text["text"] = "/" + str(self.max_page)
        if keep_pos:
            self.change_page(self.page)
        else:
            self.change_page(1)

    def current_filter(self, images):
        images = self.filter_images(images, self.filter1_exp, negative=self.filter_frame.filter1_neg.get())
        images = self.filter_images(images, self.filter2_exp, negative=self.filter_frame.filter2_neg.get())
        images = self.filter_images(images, self.filter3_exp, negative=self.filter_frame.filter3_neg.get())
        images = self.sort_images(images, self.sort_text)
        return images

    def meta_tags_filter(self, images):
        # オプションを取得
        enable_rm_filter = self.filter_frame.option_rm.get()
        enable_dup_filter = self.filter_frame.option_dup.get()
        # 各タグに作用させてTrueだったら残す用の関数
        if enable_rm_filter == 1:
            # REMOVEDをはじく
            def filter_rem(tags): return not any(re_removed.match(tag) for tag in tags)
        elif enable_rm_filter == 2:
            # REMOVEDを残す
            def filter_rem(tags): return any(re_removed.match(tag) for tag in tags)
        else:
            def filter_rem(tags): return True
        if enable_dup_filter == 1:
            # REMOVED_DUP付きを弾く
            def filter_dup(tags): return not any(re_remove_duplicated.match(tag) for tag in tags)
        elif enable_dup_filter == 2:
            # DUP_GROUP付きを残す
            def filter_dup(tags): return any(re_duplicated.match(tag) for tag in tags)
        else:
            def filter_dup(tags): return True
        # 各tagにfilter作用させた結果を評価する。
        images = [image for image in images if filter_rem(image.metatags) and filter_dup(image.metatags)]

        # DUP_GROUPでソートする
        if enable_rm_filter != 1 and enable_dup_filter == 2:
            # DUP_GROUPにマッチする要素がある前提
            images = sorted(images, key=lambda image: [tag for tag in image.metatags if re_duplicated.match(tag)][0])
        return images

    def detect_duplicated(self, thresh=0.97):
        # 全画像を対象にする
        images = self.data.all_images
        # クロップとテキスト系は類似判定の計算から除外。かつそれらのスコアが低い方を残すようにする
        avoid_tags = ["cropped_legs", "crop_top", "text_focus", "english_text", "chinese_text", "watermark",
                      "artist_name", "copyright_name", "twitter_username", "character_name"]
        avoid_tag_indexes = np.array([TaggedImage.all_tags.index(tag) for tag in avoid_tags])
        if thresh is None or thresh == 0.0:
            return images
        if avoid_tag_indexes is None:
            avoid_tag_indexes = []
        start = time.time()

        # 重複系のmetatagはリセット
        [image.remove_search_metatags(re_dups) for image in images]

        use_tag_filter = np.ones(len(TaggedImage.all_tags), dtype=np.bool_)
        use_tag_filter[avoid_tag_indexes] = False
        # 全画像のスコア(正規化済み)をarrayにして内積を計算する
        all_scores = np.array([image.scores[use_tag_filter]/np.linalg.norm(image.scores[use_tag_filter]) for image in images])
        sim_mat = np.dot(all_scores, all_scores.T)
        sim_mat_filter = sim_mat < thresh
        # 対角線をTrueで埋めたものをつくる
        sim_mat_filter_d = sim_mat_filter.copy()
        np.fill_diagonal(sim_mat_filter_d, True)
        # 重複ありがどれかを検出
        filter_dups = np.any(~sim_mat_filter_d, axis=1)
        # 重複それぞれから１つ抜き出す。重複部分をtrueにする
        filters_dups_unique = ~np.unique(sim_mat_filter[filter_dups], axis=0)
        # インデックスのリストを作成
        indexes_dups = [np.where(flt)[0] for flt in filters_dups_unique]

        # 重複ありのグループを作る
        dupgroup_id = 0
        for indexes in indexes_dups:
            dupgroup = [f"DUP_GROUP:{dupgroup_id}"]
            # 採用する画像を決める用
            avoid_score_sum_comp = 10000000000000.0  # avoid_tagsの合計スコア最小
            avoid_score_min_index = None  # avoid_tagsの合計スコア最小のやつのインデックス
            for index in indexes:
                # タググループを付与
                images[index].add_metatags(dupgroup)
                # 類似度を付与
                images[index].add_metatags(f"SIMILARITY:{sim_mat[indexes[0],index]:.3f}")
                # 採用する画像を決める
                # 全部に削除をつけてから、avoid_tagsの合計スコアが最小のものは削除取り消し
                images[index].add_metatags("REMOVED_DUP")
                avoid_score_sum = np.sum(images[index].scores[avoid_tag_indexes])
                if avoid_score_sum < avoid_score_sum_comp:
                    avoid_score_min_index = index
                    avoid_score_sum_comp = avoid_score_sum
            images[avoid_score_min_index].remove_metatags("REMOVED_DUP")
            dupgroup_id += 1
        end = time.time()
        print(f"Detection took {end-start:.6f} sec")
        # 表示を更新
        self.refresh_images(all=True)

    def filter_images(self, images, filter_exp, negative=False):
        if images is None:
            return None
        if filter_exp is None:
            return images
        if filter_exp == "":
            return images
        # フィルター記法：タグ名は{}でくくる。
        # {}内は正規表現。複数マッチしたら平均
        cmd = filter_exp
        # imageに対してcmdの評価結果を返す。

        def eval_filter(image, cmd, indexes_list, val_str):
            # 複数マッチした際は最大値をとる。合計とか平均とかでもいいかも
            compare_fn = max
            # タグのスコアを取得
            val = [compare_fn(image.scores[indexes]) for indexes in indexes_list]
            # 評価。cmdに文字列として埋め込んだval_str[x]とval[x]を対応させる
            return eval(cmd, {val_str: val})

        indexes_list = []
        val_str = "val"
        # 入力文字列が不正のエラーを避ける
        try:
            # タグの表記({tag})毎に、それにマッチするタグインデックスを取得する
            for i, match in enumerate(re.finditer("{([^}]*)}", filter_exp)):
                val_str_index = f"{val_str}[{i}]"
                tag_exp = match.group(1)
                tagindexes = [tag[0] for tag in enumerate(TaggedImage.all_tags) if re.fullmatch(tag_exp, tag[1])]
                # マッチするタグなければ中断
                if len(tagindexes) < 1:
                    print(f"No tag matches to: {tag_exp}")
                    return None
                indexes_list.append(tagindexes)
                cmd = re.sub("{" + re.escape(tag_exp) + "}", val_str_index, cmd)
                tprint(f"tag {tag_exp} matches {[TaggedImage.all_tags[i] for i in tagindexes]}")
            # cmdが空になったらそのまま帰す
            if cmd == "":
                return images
            temp_images = []
            tprint(f"filter exp:{cmd}")
            temp_images = [image for image in images if eval_filter(image, cmd, indexes_list, val_str) ^ negative]
            tprint(f"filtered image: {len(temp_images)}")
            return temp_images
        # エラーが出たらNone
        except re.error as e:
            print(e)
            return None
        except NameError as e:
            print(e)
            return None

    def sort_images(self, images, sort_text):
        if images is None:
            return None
        if sort_text in TaggedImage.all_tags:
            images = sorted(images, key=lambda image: image.get_score(sort_text), reverse=self.filter_frame.order.get())
        return images

    def change_page(self, page):
        if not isinstance(page, int):
            return
        if page < 1:
            page = 1
        if self.max_page < page:
            page = self.max_page
        self.page = page
        self.update_gallery()

    def on_images_event(self, event, fn, index):
        tprint(event)
        # イベント判定: ホイール
        if not isinstance(event.num, int) and isinstance(event.delta, int) and event.delta != 0:
            # selectedのNone避け
            if self.gallery_frame.selected is None:
                return
            if event.delta > 0:
                tprint("wheel +")
                self.change_selection(self.gallery_frame.selected - 1)
            if event.delta < 0:
                tprint("wheel -")
                self.change_selection(self.gallery_frame.selected + 1)
        # イベント判定: クリック
        elif isinstance(event.num, int) and event.num == 1:
            tprint("click")
            self.change_selection(index)
        # selectedのNone避け
        if self.gallery_frame.selected is None:
            return
        # 選択されている画像をとってきてfnを実行
        image_index = (self.page - 1) * self.gallery_frame.num_view + self.gallery_frame.selected
        images = list(self.filtered_images)
        image = images[image_index]
        self.data.set_selected_image(image)
        fn()

    def change_selection(self, index):
        # ページ切り替え判定(後)
        max_index = len(self.filtered_images)
        if index >= self.gallery_frame.num_view:
            if self.page == self.max_page:
                return
            self.change_page(self.page + 1)
            index_to = 0
        # ページ切り替え判定(前)
        elif index < 0:
            if self.page <= 1:
                return
            self.change_page(self.page - 1)
            index_to = self.gallery_frame.num_view - 1
        else:
            index_to = index
        if self.page == self.max_page and index_to >= max_index % self.gallery_frame.num_view:
            index_to = max_index % self.gallery_frame.num_view - 1
        self.gallery_frame.set_select(index_to)

    def bind_on_images(self, fn):
        self.gallery_frame.bind_on_images(lambda event, index: self.on_images_event(event, fn, index))


class ImageFilterUIFrame(tk.Frame):
    def __init__(self, master, **args) -> None:
        super().__init__(master, **args)
        vcmd_isnum = self.register(valid_isnum)
        self.order = tk.BooleanVar(value=True)
        # ソート
        self.sort_frame = tk.Frame(self)
        self.sort_txt = tk.Entry(self.sort_frame, width=20)
        self.sort_txt.pack(side=tk.LEFT)
        self.sort_button = tk.Button(self.sort_frame, text="Sort")
        self.sort_button.pack(side=tk.LEFT)
        # order: 降順=True、昇順=False
        self.order_ascend_button = tk.Radiobutton(self.sort_frame, text="Ascend", variable=self.order, value=False)
        self.order_ascend_button.pack(side=tk.LEFT)
        self.order_descend_button = tk.Radiobutton(self.sort_frame, text="Descend", variable=self.order, value=True)
        self.order_descend_button.pack(side=tk.LEFT)
        # ページ
        self.prev_button = tk.Button(self.sort_frame, text="Prev")
        self.prev_button.pack(side=tk.LEFT)
        self.page_text = tk.Entry(self.sort_frame, validate="key", validatecommand=(vcmd_isnum, '%S'), width=5)
        self.set_page_num(1)
        self.page_text.pack(side=tk.LEFT)
        self.maxpage_text = tk.Label(self.sort_frame)
        self.maxpage_text.pack(side=tk.LEFT)
        self.next_button = tk.Button(self.sort_frame, text="Next")
        self.next_button.pack(side=tk.LEFT)
        self.sort_frame.pack(side=tk.TOP)

        # フィルター1
        self.filter1_frame = tk.Frame(self)
        self.filter1_txt = tk.Entry(self.filter1_frame, width=60)
        self.filter1_txt.pack(side=tk.LEFT, anchor=tk.W)
        self.filter1_button = tk.Button(self.filter1_frame, text="Filter1")
        self.filter1_button.pack(side=tk.LEFT)
        self.filter1_neg = tk.BooleanVar(value=False)
        self.filter1_neg_cbt = tk.Checkbutton(self.filter1_frame, text="negative", variable=self.filter1_neg)
        self.filter1_neg_cbt.pack(side=tk.LEFT)
        self.filter1_frame.pack(side=tk.TOP)
        # フィルター2
        self.filter2_frame = tk.Frame(self)
        self.filter2_txt = tk.Entry(self.filter2_frame, width=60)
        self.filter2_txt.pack(side=tk.LEFT, anchor=tk.W)
        self.filter2_button = tk.Button(self.filter2_frame, text="Filter2")
        self.filter2_button.pack(side=tk.LEFT)
        self.filter2_neg = tk.BooleanVar(value=False)
        self.filter2_neg_cbt = tk.Checkbutton(self.filter2_frame, text="negative", variable=self.filter2_neg)
        self.filter2_neg_cbt.pack(side=tk.LEFT)
        self.filter2_frame.pack(side=tk.TOP)
        # フィルター3
        self.filter3_frame = tk.Frame(self)
        self.filter3_txt = tk.Entry(self.filter3_frame, width=60)
        self.filter3_txt.pack(side=tk.LEFT, anchor=tk.W)
        self.filter3_button = tk.Button(self.filter3_frame, text="Filter3")
        self.filter3_button.pack(side=tk.LEFT)
        self.filter3_neg = tk.BooleanVar(value=False)
        self.filter3_neg_cbt = tk.Checkbutton(self.filter3_frame, text="negative", variable=self.filter3_neg)
        self.filter3_neg_cbt.pack(side=tk.LEFT)
        self.filter3_frame.pack(side=tk.TOP)
        # フィルタオプション
        self.option_frame = tk.Frame(self)
        self.option_lbl = tk.Label(self, text="▼Display Option")
        self.option_lbl.bind("<Button-1>", lambda event: toggle_object(self.option_frame))
        self.option_lbl.pack(side=tk.TOP)
        self.option_apply_btn = tk.Button(self.option_frame, text="Apply")
        # REMOVEDの有無
        self.option_rm = tk.IntVar(value=1)
        self.option_rm_lbl = tk.Label(self.option_frame, text="Filter Removed")
        self.option_rm_cbn = tk.Radiobutton(self.option_frame, text="Enable", variable=self.option_rm, value=1)
        self.option_rm_inv_cbn = tk.Radiobutton(self.option_frame, text="Invert", variable=self.option_rm, value=2)
        self.option_rm_off_cbn = tk.Radiobutton(self.option_frame, text="Disable", variable=self.option_rm, value=0)
        # DUPLICATEの有無
        self.option_dup = tk.IntVar(value=1)
        self.option_dup_lbl = tk.Label(self.option_frame, text="Filter Duplicated")
        self.option_dup_cbn = tk.Radiobutton(self.option_frame, text="Enable", variable=self.option_dup, value=1)
        self.option_dup_inv_cbn = tk.Radiobutton(self.option_frame, text="Invert", variable=self.option_dup, value=2)
        self.option_dup_off_cbn = tk.Radiobutton(self.option_frame, text="Disable", variable=self.option_dup, value=0)
        self.option_apply_btn.grid(row=0, column=0, columnspan=2)
        self.option_rm_lbl.grid(row=1, column=0, sticky=tk.W)
        self.option_rm_cbn.grid(row=1, column=1, sticky=tk.W)
        self.option_rm_inv_cbn.grid(row=1, column=2, sticky=tk.W)
        self.option_rm_off_cbn.grid(row=1, column=3, sticky=tk.W)
        self.option_dup_lbl.grid(row=2, column=0, sticky=tk.W)
        self.option_dup_cbn.grid(row=2, column=1, sticky=tk.W)
        self.option_dup_inv_cbn.grid(row=2, column=2, sticky=tk.W)
        self.option_dup_off_cbn.grid(row=2, column=3, sticky=tk.W)
        # 類似画像検出
        self.detect_dup_frame = tk.Frame(self.option_frame)
        self.detect_dup_button = tk.Button(self.detect_dup_frame, text="DetectDuplicated")
        self.detect_dup_thresh_lbl = tk.Label(self.detect_dup_frame, text="Threshold:")
        self.detect_dup_thresh = tk.Entry(self.detect_dup_frame, width=5)
        self.detect_dup_thresh.insert(0, "0.97")
        self.detect_dup_button.pack(side=tk.LEFT)
        self.detect_dup_thresh_lbl.pack(side=tk.LEFT)
        self.detect_dup_thresh.pack(side=tk.LEFT)
        self.detect_dup_frame.grid(row=3, column=0, columnspan=3, sticky=tk.W)

    def set_page_num(self, num):
        if not isinstance(num, int):
            return
        self.page_text.delete(0, tk.END)
        self.page_text.insert(tk.END, num)

    def get_page_num(self):
        num = self.page_text.get()
        if re.match(re.compile('[0-9]+'), num):
            return int(num)
        else:
            return None


class SelectionViewerFrame(tk.Frame):
    def __init__(self, master, data, imsize=(500, 500), **args) -> None:
        self.data = data
        self.image = None
        self.frame_img = None
        self.imsize = imsize
        super().__init__(master, **args)
        self.frame_img_wrapper = tk.Frame(self, width=self.imsize[0], height=self.imsize[1], background="gray")
        self.frame_img_wrapper.propagate(False)
        self.frame_img_wrapper.pack(side=tk.TOP)
        self.frame_fixedtags = tk.Text(self, width=70, height=10, undo=True, wrap=tk.WORD)
        self.frame_fixedtags.propagate(False)
        self.frame_fixedtags.pack(side=tk.TOP)
        self.frame_metatags = tk.Text(self, width=70, height=1, undo=True, wrap=tk.WORD)
        self.frame_metatags.propagate(False)
        self.frame_metatags.pack(side=tk.TOP)
        self.set_fixed_tags_btn = tk.Button(self, text="ModifyTags", command=self.modify_tags)
        self.set_fixed_tags_btn.pack(side=tk.TOP)

    def set_to_selected(self):
        self.set_image(self.data.selected_image)

    def set_image(self, image):
        self.clear_image()
        self.image = image
        if image is not None:
            img = image.get_img()
            img = keep_resize(img, self.imsize, 0.0, Image.Resampling.LANCZOS)
            photo = ImageTk.PhotoImage(img)
            self.frame_img = tk.Label(self.frame_img_wrapper, image=photo)
            self.frame_img.image = photo
            self.frame_img.pack()
            self.reload_tags()

    def clear_image(self):
        self.image = None
        if self.frame_img is not None:
            self.frame_img.destroy()
            self.frame_img = None

    def modify_tags(self):
        self.image.set_fixedtags(re_separater.sub(",", self.frame_fixedtags.get(0., tk.END).strip(" ,\t\n\r")).split(","))
        self.image.set_metatags(re_separater.sub(",", self.frame_metatags.get(0., tk.END).strip(" ,\t\n\r")).split(","))
        self.reload_tags()

    def reload_tags(self):
        self.frame_fixedtags.delete(0., tk.END)
        self.frame_fixedtags.insert(0., ", ".join(self.image.fixedtags).strip(" ,\t\n\r"))
        self.frame_metatags.delete(0., tk.END)
        self.frame_metatags.insert(0., ", ".join(self.image.metatags).strip(" ,\t\n\r"))


class TagItemFrame(tk.Frame):
    def on_event(x, y): return x

    def __init__(self, master, imsize, index, **args) -> None:
        self.highlight = False
        self.index = index
        self.width, self.height = imsize
        score_width = 4
        super().__init__(master, width=self.width, height=self.height, **args)
        self.propagate(False)
        self.label_tag = tk.Label(self, width=(self.width//2 - score_width), anchor=tk.W)
        self.label_tag.propagate(False)
        self.label_score = tk.Label(self, width=score_width, anchor=tk.E, background="gray80")
        self.label_score.propagate(False)
        self.label_tag.grid(row=0, column=0)
        self.label_score.grid(row=0, column=1)

    def set_tag(self, tagscore):
        if tagscore is None:
            self.clear_tag()
            return
        tag, score = tagscore
        score = f"{score:.3f}"
        self.label_tag["text"] = tag
        self.label_score["text"] = score

    def update_frame(self):
        # if self.highlight:
        #     self.configure(background="gray")
        # else:
        #     self.configure(background="white")
        self.update()

    def set_highlight(self, highlight):
        if highlight is None:
            self.highlight = not self.highlight
        else:
            self.highlight = highlight
        self.update_frame()

    def bind_keys(self):
        # self.bind("<Button-1>", lambda event: __class__.on_event(event, self.index))
        self.bind("<MouseWheel>", lambda event: __class__.on_event(event, self.index))
        self.label_tag.bind("<Button-1>", lambda event: __class__.on_event(event, self.index))
        self.label_tag.bind("<MouseWheel>", lambda event: __class__.on_event(event, self.index))
        # self.label_score.bind("<Button-1>", lambda event: __class__.on_event(event, self.index))
        self.label_score.bind("<MouseWheel>", lambda event: __class__.on_event(event, self.index))

    def clear_tag(self):
        self.label_tag["text"] = ""
        self.label_score["text"] = ""


class TagsGalleryFrame(tk.Frame):
    def __init__(self, master, row=30, column=3, **args) -> None:
        self.selected = None
        self.column = column
        self.num_view = column * row
        super().__init__(master, **args, background="white")
        imsize = (30, 10)
        # 子フレームを作成
        self.frames = [TagItemFrame(self, imsize=imsize, index=i) for i in range(column*row)]
        for i, frame in enumerate(self.frames):
            frame.grid(row=i//column, column=i % column)

    def set_tags(self, tagscores):
        tagscores = list(tagscores)[0:self.num_view]
        # 各フレームに画像を設定
        for i, frame in enumerate(self.frames):
            if i < len(tagscores):
                frame.set_tag(tagscores[i])
            else:
                frame.set_tag(None)
        self.update()

    def set_select(self, index):
        # 選択されたindexを保存し、ハイライト判定
        if self.selected is not None:
            self.frames[self.selected].set_highlight(False)
            self.frames[self.selected].update_frame()
        if index is not None:
            self.frames[index].set_highlight(True)
        self.selected = index

    def bind_on_tags(self, fn):
        # クラスアトリビュート経由でfnをバインドする
        TagItemFrame.on_event = fn
        for frame in self.frames:
            frame.bind_keys()


class TagUIFrame(tk.Frame):
    def __init__(self, master, **args) -> None:
        super().__init__(master, **args)
        vcmd_isnum = self.register(valid_isnum)
        self.order = tk.IntVar(value=2)

        self.filterfram = tk.Frame(self)
        row = 0
        # テキストフィルタ1
        self.text1lbl = tk.Label(self.filterfram, text="Text Filter1")
        self.text1lbl.grid(row=row, column=0)
        self.text1txt = tk.Entry(self.filterfram, width=40)
        self.text1txt.grid(row=row, column=1)
        self.text1filter_button = tk.Button(self.filterfram, text="Filter")
        self.text1filter_button.grid(row=row, column=2)
        self.text1_neg = tk.BooleanVar(value=False)
        self.text1_neg_ckb = tk.Checkbutton(self.filterfram, text="Negative", variable=self.text1_neg)
        self.text1_neg_ckb.grid(row=row, column=3)
        # テキストフィルタ2
        row = 1
        self.text2lbl = tk.Label(self.filterfram, text="Text Filter2")
        self.text2lbl.grid(row=row, column=0)
        self.text2txt = tk.Entry(self.filterfram, width=40)
        self.text2txt.grid(row=row, column=1)
        self.text2filter_button = tk.Button(self.filterfram, text="Filter")
        self.text2filter_button.grid(row=row, column=2)
        self.text2_neg = tk.BooleanVar(value=False)
        self.text2_neg_ckb = tk.Checkbutton(self.filterfram, text="Negative", variable=self.text2_neg)
        self.text2_neg_ckb.grid(row=row, column=3)
        # テキストフィルタ3
        row = 2
        self.text3lbl = tk.Label(self.filterfram, text="Text Filter3")
        self.text3lbl.grid(row=row, column=0)
        self.text3txt = tk.Entry(self.filterfram, width=40)
        self.text3txt.grid(row=row, column=1)
        self.text3filter_button = tk.Button(self.filterfram, text="Filter")
        self.text3filter_button.grid(row=row, column=2)
        self.text3_neg = tk.BooleanVar(value=False)
        self.text3_neg_ckb = tk.Checkbutton(self.filterfram, text="Negative", variable=self.text3_neg)
        self.text3_neg_ckb.grid(row=row, column=3)

        # スコアフィルタ
        row = 3
        self.scorelbl = tk.Label(self.filterfram, text="Score Filter")
        self.scorelbl.grid(row=row, column=0)
        self.scoretxt = tk.Entry(self.filterfram, width=40)
        self.scoretxt.grid(row=row, column=1)
        self.scorefilter_button = tk.Button(self.filterfram, text="Filter")
        self.scorefilter_button.grid(row=row, column=2)
        self.score_neg = tk.BooleanVar(value=False)
        self.score_neg_ckb = tk.Checkbutton(self.filterfram, text="Less Than", variable=self.score_neg)
        self.score_neg_ckb.grid(row=row, column=3)

        # ランクフィルタ
        row = 4
        self.ranklbl = tk.Label(self.filterfram, text="Rank Filter")
        self.ranklbl.grid(row=row, column=0)
        self.ranktxt = tk.Entry(self.filterfram, width=40, validate="key", validatecommand=(vcmd_isnum, '%S'))
        self.ranktxt.grid(row=row, column=1)
        self.rankfilter_button = tk.Button(self.filterfram, text="Filter")
        self.rankfilter_button.grid(row=row, column=2)
        self.filterfram.pack(side=tk.TOP)
        # order: アルファベット昇順=0, アルファベット降順=1, スコア昇順=2,スコア降順=3
        self.orderframe = tk.Frame(self)
        row = 0
        self.textsortlbl = tk.Label(self.orderframe, text="Sort by Alph:")
        self.textsortlbl.grid(row=row, column=0)
        self.textorder_ascend_button = tk.Radiobutton(self.orderframe, text="Ascend", variable=self.order, value=0)
        self.textorder_ascend_button.grid(row=row, column=1)
        self.textorder_descend_button = tk.Radiobutton(self.orderframe, text="Descend", variable=self.order, value=1)
        self.textorder_descend_button.grid(row=row, column=2)
        row = 1
        self.scoresortlbl = tk.Label(self.orderframe, text="Sort by Score:")
        self.scoresortlbl.grid(row=row, column=0)
        self.scoreorder_ascend_button = tk.Radiobutton(self.orderframe, text="Ascend", variable=self.order, value=2)
        self.scoreorder_ascend_button.grid(row=row, column=1)
        self.scoreorder_descend_button = tk.Radiobutton(self.orderframe, text="Descend", variable=self.order, value=3)
        self.scoreorder_descend_button.grid(row=row, column=2)
        self.orderframe.pack(side=tk.TOP)


class TagsViewerFrame(tk.Frame):
    def __init__(self, master, data, **args) -> None:
        self.data = data
        self.offset_rows = 0
        # self.filter1_exp = ""
        # self.filter2_exp = ""
        # self.filter3_exp = ""
        self.index_filter1 = np.bool_(True)
        self.index_filter2 = np.bool_(True)
        self.index_filter3 = np.bool_(True)
        self.index_filter = np.bool_(True)
        self.score_thresh = np.float32(0.0)
        self.selected_scores = None
        self.filter_rank = 0
        self.tags_dict = {}
        self.max_row = 0
        # self.width = width
        # self.height = height
        super().__init__(master, **args)
        # self.propagate(False)
        self.tag_gallery = TagsGalleryFrame(self)
        self.tag_gallery.pack()
        self.bind_on_images()

        # filterのボタン設定
        self.filter_frame = TagUIFrame(self)
        self.filter_frame.text1filter_button.config(command=lambda: self.refresh_gallery(filter1_exp=self.filter_frame.text1txt.get()))
        self.filter_frame.text1txt.bind('<Return>', lambda event: self.refresh_gallery(filter1_exp=self.filter_frame.text1txt.get()))
        self.filter_frame.text1_neg_ckb.config(command=lambda: self.refresh_gallery(filter1_exp=self.filter_frame.text1txt.get()))
        self.filter_frame.text2filter_button.config(command=lambda: self.refresh_gallery(filter2_exp=self.filter_frame.text2txt.get()))
        self.filter_frame.text2txt.bind('<Return>', lambda event: self.refresh_gallery(filter2_exp=self.filter_frame.text2txt.get()))
        self.filter_frame.text2_neg_ckb.config(command=lambda: self.refresh_gallery(filter2_exp=self.filter_frame.text2txt.get()))
        self.filter_frame.text3filter_button.config(command=lambda: self.refresh_gallery(filter3_exp=self.filter_frame.text3txt.get()))
        self.filter_frame.text3txt.bind('<Return>', lambda event: self.refresh_gallery(filter3_exp=self.filter_frame.text3txt.get()))
        self.filter_frame.text3_neg_ckb.config(command=lambda: self.refresh_gallery(filter3_exp=self.filter_frame.text3txt.get()))
        self.filter_frame.scorefilter_button.config(command=lambda: self.refresh_gallery(score_thresh=str2npf(self.filter_frame.scoretxt.get())))
        self.filter_frame.scoretxt.bind('<Return>', lambda event: self.refresh_gallery(score_thresh=str2npf(self.filter_frame.scoretxt.get())))
        self.filter_frame.score_neg_ckb.config(command=lambda: self.refresh_gallery(score_thresh=str2npf(self.filter_frame.scoretxt.get())))
        self.filter_frame.rankfilter_button.config(command=lambda: self.refresh_gallery(filter_rank=str2int(self.filter_frame.ranktxt.get())))
        self.filter_frame.ranktxt.bind('<Return>', lambda event: self.refresh_gallery(filter_rank=str2int(self.filter_frame.ranktxt.get())))
        self.filter_frame.textorder_ascend_button.config(command=self.refresh_gallery)
        self.filter_frame.textorder_descend_button.config(command=self.refresh_gallery)
        self.filter_frame.scoreorder_ascend_button.config(command=self.refresh_gallery)
        self.filter_frame.scoreorder_descend_button.config(command=self.refresh_gallery)
        self.refresh_gallery()
        self.filter_frame.pack()

    def update_gallery(self) -> None:
        n = self.offset_rows*self.tag_gallery.column
        tagscores = [(TaggedImage.all_tags[index], self.selected_scores[index])
                     for i, index in enumerate(self.filtered_indexes) if n <= i < n+self.tag_gallery.num_view]
        self.tag_gallery.set_tags(tagscores)
        return self.update()

    def refresh_gallery(self, filter1_exp=None, filter2_exp=None, filter3_exp=None, score_thresh=None, filter_rank=None) -> None:
        # if filter1_exp is None:  # デフォルトでは元のものを使う
        #     filter1_exp = self.filter1_exp
        # if filter2_exp is None:  # デフォルトでは元のものを使う
        #     filter2_exp = self.filter2_exp
        # if filter3_exp is None:  # デフォルトでは元のものを使う
        #     filter3_exp = self.filter3_exp
        if score_thresh is None:  # デフォルトでは元のものを使う
            score_thresh = self.score_thresh
        if filter_rank is None:  # デフォルトでは元のものを使う
            filter_rank = self.filter_rank
        if self.apply_filter_on_tags(filter1_exp=filter1_exp, filter2_exp=filter2_exp, filter3_exp=filter3_exp, score_thresh=score_thresh, filter_rank=filter_rank) is None:
            return
        self.offset_rows = 0
        self.max_row = (len(self.filtered_indexes)-1)//self.tag_gallery.column
        self.tag_gallery.set_select(None)
        self.offset_gallery(0)
        return self.update_gallery()

    def update_exp_filters(self, filter1_exp=None, filter2_exp=None, filter3_exp=None):
        tag_indexes = self.filter_tag_indexes_by_exp(filter1_exp, negative=self.filter_frame.text1_neg.get())
        if tag_indexes is not None:
            self.index_filter1 = tag_indexes
        tag_indexes = self.filter_tag_indexes_by_exp(filter2_exp, negative=self.filter_frame.text2_neg.get())
        if tag_indexes is not None:
            self.index_filter2 = tag_indexes
        tag_indexes = self.filter_tag_indexes_by_exp(filter3_exp, negative=self.filter_frame.text3_neg.get())
        if tag_indexes is not None:
            self.index_filter3 = tag_indexes
        self.index_filter = self.index_filter1 & self.index_filter2 & self.index_filter3

    def apply_filter_on_tags(self, filter1_exp, filter2_exp, filter3_exp, score_thresh, filter_rank):
        # self変数のアップデート判定
        if score_thresh is not None:
            self.score_thresh = score_thresh
        if filter_rank is not None:
            self.filter_rank = filter_rank
        # textフィルタをアップデート
        self.update_exp_filters(filter1_exp=filter1_exp, filter2_exp=filter2_exp, filter3_exp=filter3_exp)
        # フィルタを適用
        filtered_indexes = self.current_filter(self.selected_scores)
        if filtered_indexes is not None:
            self.filtered_indexes = filtered_indexes
        return filtered_indexes

    def current_filter(self, scores):
        index_filter = self.filter_tag_indexes_by_score(self.index_filter, scores, self.score_thresh,
                                                        less_than=self.filter_frame.score_neg.get())
        filtered_indexes = self.sort_tags(index_filter, scores, order=self.filter_frame.order.get(), filter_rank=self.filter_rank)
        return filtered_indexes

    def filter_tag_indexes_by_exp(self, filter_exp, negative=False):
        # npのone shot boolを返す
        # 入力がNoneならNoneを返す
        if filter_exp is None:
            return None
        try:  # reのエラーを回避
            # negative がTrueなら反転
            tags_indexes = np.array([bool(re.search(filter_exp, tag)) ^ negative for tag in TaggedImage.all_tags], dtype=np.bool_)
            tprint(f"filter exp:{filter_exp}")
            return tags_indexes
        except re.error as e:
            print("re.error:", e)
            return None

    def filter_tag_indexes_by_score(self, index_filter, scores, thresh, less_than=False):
        # npのone shot boolを返す
        # スコアが空かNoneならNoneを返す
        if thresh is None or scores is None:
            return None
        tprint(f"filter score:{thresh}")
        if index_filter is None:
            index_filter = np.bool_(True)
        # index_filterがあれば先にフィルタしてから比較
        # less_than がTrueなら反転
        return index_filter & ((scores >= thresh) ^ less_than)

    def sort_tags(self, index_filter, scores, order, filter_rank=0):
        if index_filter is None or scores is None:
            return None
        filtered_indexes = TaggedImage.all_indexes[index_filter]
        if order in [0, 1]:
            # アルファベット順
            filtered_target = TaggedImage.all_alph_order[index_filter]
            sorted_indexes = filtered_indexes[np.argsort(filtered_target)]
            # 1なら逆順
            if order == 1:
                sorted_indexes = sorted_indexes[::-1]
        elif order in [2, 3]:
            # スコア順
            filtered_target = scores[index_filter]
            sorted_indexes = filtered_indexes[np.argsort(filtered_target)]
            # 1なら逆順
            if order == 2:
                sorted_indexes = sorted_indexes[::-1]
        if filter_rank is None or filter_rank == 0:
            return sorted_indexes
        else:
            return sorted_indexes[0:filter_rank]

    def set_to_selected(self):
        self.selected_scores = self.data.selected_image.scores
        self.refresh_gallery()

    def offset_gallery(self, n_row):
        self.offset_rows = max(0, min(self.max_row, n_row))

    def on_tags_event(self, event, index):
        tprint(event)
        # イベント判定: ホイール
        wheel_tick = 3
        if not isinstance(event.num, int) and isinstance(event.delta, int) and event.delta != 0:
            if event.delta > 0:
                tprint("wheel +")
                self.offset_gallery(self.offset_rows - wheel_tick)
            if event.delta < 0:
                tprint("wheel -")
                self.offset_gallery(self.offset_rows + wheel_tick)
        # イベント判定: クリック
        elif isinstance(event.num, int) and event.num == 1:
            tprint("click")
            w = event.widget
            if isinstance(w, tk.Label) and len(w["text"]) > 0:
                pyperclip.copy(w["text"])
                w["relief"] = tk.SUNKEN
                w.after(200, lambda: w.configure(relief=tk.FLAT))
        self.update_gallery()

    def bind_on_images(self):
        self.tag_gallery.bind_on_tags(self.on_tags_event)


class FixedTagEditter(tk.Frame):
    def __init__(self, master, data: DataManager, **args) -> None:
        self.data = data
        self.master = master
        self.offset_rows = 0
        self.filter_exp = ""
        self.filter_score = 0.0
        self.tags = {}
        self.max_row = 0
        super().__init__(master, **args)
        # 対象選択ボタン
        self.radio_frame = tk.Frame(self)
        self.label = tk.Label(self.radio_frame, text="Change For")
        self.label.pack(side=tk.LEFT, anchor=tk.CENTER)
        self.images_for = tk.IntVar(value=1)
        self.images_for_sel_radio = tk.Radiobutton(self.radio_frame, text="Selected", variable=self.images_for, value=0)
        self.images_for_displayed_radio = tk.Radiobutton(self.radio_frame, text="Displayed", variable=self.images_for, value=1)
        self.images_for_all_radio = tk.Radiobutton(self.radio_frame, text="All", variable=self.images_for, value=2)
        self.images_for_sel_radio.pack(side=tk.LEFT, anchor=tk.CENTER)
        self.images_for_displayed_radio.pack(side=tk.LEFT, anchor=tk.CENTER)
        self.images_for_all_radio.pack(side=tk.LEFT, anchor=tk.CENTER)
        self.radio_frame.pack(side=tk.TOP)
        # 実行ボタン
        self.button_frame = tk.Frame(self)
        self.add_tags_btn = tk.Button(self.button_frame, text="AddFilteredTags", command=self.add_displayed_fixedtags)
        self.remove_tags_btn = tk.Button(self.button_frame, text="RemoveFilteredTags", command=self.remove_displayed_fixedtags)
        self.uniq_tags_btn = tk.Button(self.button_frame, text="UniquifyTags", command=self.uniquefy_fixedtags)
        self.add_tags_btn.grid(row=0, column=0, padx=5)
        self.remove_tags_btn.grid(row=0, column=1, padx=5)
        self.uniq_tags_btn.grid(row=0, column=2, padx=5)
        self.button_frame.pack(side=tk.TOP)
        # 置換フレーム
        self.subst_frame = tk.Frame(self)
        self.subst_exp_label = tk.Label(self.subst_frame, text="Subst tags")
        self.subst_exp = tk.Entry(self.subst_frame, width=10)
        self.subst_to_label = tk.Label(self.subst_frame, text="to")
        self.subst_to = tk.Entry(self.subst_frame, width=10)
        self.subst_btn = tk.Button(self.subst_frame, text="Subst", command=self.subst_fixedtags)
        self.subst_exp_label.pack(side=tk.LEFT)
        self.subst_exp.pack(side=tk.LEFT)
        self.subst_to_label.pack(side=tk.LEFT)
        self.subst_to.pack(side=tk.LEFT)
        self.subst_btn.pack(side=tk.LEFT)
        self.subst_frame.pack(side=tk.TOP)
        # REMOVE
        self.rm_button_frame = tk.Frame(self)
        self.remove_image_btn = tk.Button(self.rm_button_frame, text="RemoveImage", command=self.add_to_removed)
        self.restore_image_btn = tk.Button(self.rm_button_frame, text="RestoreImage", command=self.restore_removed)
        self.remove_image_btn.grid(row=0, column=0, padx=5)
        self.restore_image_btn.grid(row=0, column=1, padx=5)
        self.rm_button_frame.pack(side=tk.TOP)

    def get_target_images(self):
        # master経由でfilterd_gallery_frameから表示中の画像を取得
        if self.images_for.get() == 0:
            # 選択中画像。 master経由でselection_viewer_frameから画像を取得
            images = [self.master.selection_viewer_frame.image]
        elif self.images_for.get() == 1:
            # フィルター後画像。 master経由でfilterd_gallery_frameから表示中の画像を取得
            images = self.master.filterd_gallery_frame.filtered_images
        elif self.images_for.get() == 2:
            # 全画像。 master経由でfilterd_gallery_frameから表示中の画像を取得
            images = self.data.all_images
        else:
            return
        return images

    def add_to_removed(self):
        images = self.get_target_images()
        for image in images:
            image.add_metatags("REMOVED_BY_USER")
        self.master.selection_viewer_frame.reload_tags()
        self.master.filterd_gallery_frame.refresh_images(all=True, keep_pos=True)

    def restore_removed(self):
        images = self.get_target_images()
        for image in images:
            image.remove_metatags("REMOVED_BY_USER")
        self.master.selection_viewer_frame.reload_tags()
        self.master.filterd_gallery_frame.refresh_images(all=True, keep_pos=True)

    def subst_fixedtags(self):
        match_item = re.compile(self.subst_exp.get())
        sub_to = self.subst_to.get()
        images = self.get_target_images()
        for image in images:
            image.subst_fixedtags(match_item, sub_to)
        self.master.selection_viewer_frame.reload_tags()

    def add_displayed_fixedtags(self):
        images = self.get_target_images()
        for image in images:
            # master経由でtags_viewer_frameからタグフィルタを適用
            filtered_indexes = self.master.tags_viewer_frame.current_filter(image.scores)
            image.add_fixedtags_by_index(filtered_indexes, prepend=False, uniq=True)
        self.master.selection_viewer_frame.reload_tags()

    def remove_displayed_fixedtags(self):
        images = self.get_target_images()
        for image in images:
            # master経由でtags_viewer_frameからタグフィルタを適用
            filtered_indexes = self.master.tags_viewer_frame.current_filter(image.scores)
            image.remove_fixedtags_by_index(filtered_indexes)
        self.master.selection_viewer_frame.reload_tags()

    def uniquefy_fixedtags(self):
        images = self.get_target_images()
        for image in images:
            image.uniquefy_fixedtags()
        self.master.selection_viewer_frame.reload_tags()


class FileManageUI(tk.Frame):
    def __init__(self, master, data: DataManager, **args) -> None:
        self.data = data
        self.master = master
        super().__init__(master, **args)
        # テキストボックス
        self.images_dir = tk.StringVar()
        self.tagtexts_dir = tk.StringVar()
        self.jsons_dir = tk.StringVar()
        self.out_kohyajsons_path = tk.StringVar()
        self.out_tagtext_dir = tk.StringVar()
        input_frame = tk.Frame(self)
        images_label = tk.Label(input_frame, text="images dir:", anchor=tk.E)
        tagtexts_label = tk.Label(input_frame, text="tag texts dir:", anchor=tk.E)
        jsons_label = tk.Label(input_frame, text="jsons dir:", anchor=tk.E)
        self.images_dir_box = tk.Entry(input_frame, width=50, textvariable=self.images_dir)
        self.tagtexts_dir_box = tk.Entry(input_frame, width=50, textvariable=self.tagtexts_dir)
        self.jsons_dir_box = tk.Entry(input_frame, width=50, textvariable=self.jsons_dir)
        images_label.grid(row=0, column=0, sticky=tk.E)
        self.images_dir_box.grid(row=0, column=1)
        tagtexts_label.grid(row=1, column=0, sticky=tk.E)
        self.tagtexts_dir_box.grid(row=1, column=1)
        jsons_label.grid(row=2, column=0, sticky=tk.E)
        self.jsons_dir_box.grid(row=2, column=1)
        input_frame.pack(side=tk.TOP)
        # 各種ボタン
        self.load_btn = tk.Button(self, text="Load", command=self.load_datas)
        self.load_btn.pack(side=tk.TOP)

        self.output_frame = tk.Frame(self)
        self.output_text_box = tk.Entry(self.output_frame, width=40, textvariable=self.out_tagtext_dir)
        self.output_text_btn = tk.Button(self.output_frame, text="Dump Tags Text", command=self.write_fixed_tags)
        self.output_text_box.grid(row=0, column=0)
        self.output_text_btn.grid(row=0, column=1)

        self.output_kohya_box = tk.Entry(self.output_frame, width=40, textvariable=self.out_kohyajsons_path)
        self.output_kohya_btn = tk.Button(self.output_frame, text="Dump Json for Kohya Trainer", command=self.write_fixed_tags_kohya)
        self.output_kohya_box.grid(row=1, column=0)
        self.output_kohya_btn.grid(row=1, column=1)
        self.output_frame.pack(side=tk.TOP)

    def load_datas(self):
        self.data.load_data(images_dir=self.images_dir.get(), tagtexts_dir=self.tagtexts_dir.get(), jsons_dir=self.jsons_dir.get())
        self.master.refresh(all=True)

    def write_fixed_tags_kohya(self):
        outfile = self.out_kohyajsons_path.get()
        # REMOVEDがついている画像は対象外
        kohyajson = {image.name: {"tags": ", ".join(image.fixedtags).strip(", \t\n\r")}
                     for image in self.data.all_images if not image.search_metatags(re_removed)}
        with open(outfile, "wt", encoding='utf-8') as f:
            json.dump(kohyajson, f, indent=2)

    def write_fixed_tags(self):
        ext = ".txt"
        outdir = self.out_tagtext_dir.get()
        # REMOVEDがついている画像は対象外
        for image in (image for image in self.data.all_images if not image.search_metatags(re_removed)):
            tagstr = ", ".join(image.fixedtags).strip(", \t\n\r")
            with open(os.path.join(outdir, image.name + ext), "wt", encoding='utf-8') as f:
                f.write(tagstr)
        # 対象外の画像のリストを保存
        removed = [image.name for image in self.data.all_images if image.search_metatags(re_removed)]
        with open(os.path.join(outdir, "removed_images.txt"), "wt", encoding='utf-8') as f:
            f.write("\n".join(removed).strip("\t\n\r"))

    def write_states(self):
        # メタタグ含めて保存する
        outfile = "state.json"
        statejson = {"images_dir": self.data.images_dir, "tagtexts_dir": self.data.tagtexts_dir, "jsons_dir": self.data.jsons_dir}
        statejson["image_datas"] = {image.name: {"fixedtags": image.fixedtags, "metatags": image.metatags} for image in self.data.all_images}
        with open(outfile, "wt", encoding='utf-8') as f:
            json.dump(statejson, f, indent=2)


class ImageBrowser(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Image Browser")
        self.data = DataManager()
        self.filterd_gallery_frame = FilteredGalleryFrame(self, self.data)
        self.filterd_gallery_frame.pack(side=tk.LEFT, anchor=tk.N)
        self.tags_viewer_frame = TagsViewerFrame(self, data=self.data)
        self.tags_viewer_frame.pack(side=tk.RIGHT, anchor=tk.N)
        self.selection_viewer_frame = SelectionViewerFrame(self, data=self.data)
        self.selection_viewer_frame.pack(side=tk.TOP, anchor=tk.CENTER)
        self.fixed_tag_edditer = FixedTagEditter(self, self.data)
        self.fixed_tag_edditer.pack(side=tk.TOP, pady=10, anchor=tk.CENTER)
        self.file_manage_ui = FileManageUI(self, self.data)
        self.file_manage_ui_lbl = tk.Label(self, text="▼Files")
        self.file_manage_ui_lbl.bind("<Button-1>", lambda event: toggle_object(self.file_manage_ui))
        self.file_manage_ui_lbl.pack(side=tk.TOP, anchor=tk.CENTER)
        self.filterd_gallery_frame.bind_on_images(lambda: (self.selection_viewer_frame.set_to_selected(), self.tags_viewer_frame.set_to_selected()))

    def refresh(self, all=False):
        self.filterd_gallery_frame.refresh_images(all=all)


def main():
    browser = ImageBrowser()
    browser.mainloop()


if __name__ == "__main__":
    main()
