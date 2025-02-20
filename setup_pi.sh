#!/bin/sh

# Exit immediately if a command exits with a non-zero status
set -e

# Update and upgrade the system
sudo apt-get update -y && sudo apt-get upgrade -y

# Install Vim
sudo apt-get install vim -y

# Install Zsh
sudo apt-get install zsh -y

# Install Oh My Zsh
sh -c "$(curl -fsSL https://raw.github.com/ohmyzsh/ohmyzsh/master/tools/install.sh)"

# Change the default shell to Zsh
chsh -s $(which zsh)

# Add aliases to .zshrc
echo "\n# Custom Aliases\nalias u='cd ../;'\nalias uu='u u'\nalias uuu='u u u'\nalias uuuu='u u u u'\nalias uuuuu='u u u u u'\nalias uuuuuu='u u u u u u'" >> ~/.zshrc

# Create .vimrc in home directory and add configurations
cat <<EOL >> ~/.vimrc
" Standard vim options
set autoindent            " always set autoindenting on
set backspace=2           " allow backspacing over everything in insert mode
set cindent               " c code indenting
set diffopt=filler,iwhite " keep files synced and ignore whitespace
set expandtab             " Get rid of tabs altogether and replace with spaces
set foldcolumn=2          " set a column incase we need it
set foldlevel=0           " show contents of all folds
set foldmethod=indent     " use indent unless overridden
set guioptions-=m         " Remove menu from the gui
set guioptions-=T         " Remove toolbar
set hidden                " hide buffers instead of closing
set history=50            " keep 50 lines of command line history
set ignorecase            " Do case insensitive matching
set incsearch             " Incremental search
set laststatus=2          " always have status bar
set linebreak             " This displays long lines as wrapped at word boundries
set matchtime=10          " Time to flash the brack with showmatch
set nobackup              " Don't keep a backup file
set nocompatible          " Use Vim defaults (much better!)
set nofen                 " disable folds
set notimeout             " i like to be pokey
set ttimeout              " timeout on key-codes
set ttimeoutlen=100       " timeout on key-codes after 100ms
set ruler                 " the ruler on the bottom is useful
set scrolloff=1           " dont let the curser get too close to the edge
set shiftwidth=4          " Set indention level to be the same as softtabstop
set showcmd               " Show (partial) command in status line.
set showmatch             " Show matching brackets.
set softtabstop=4         " Why are tabs so big?  This fixes it
set textwidth=0           " Don't wrap words by default
set virtualedit=block     " let blocks be in virutal edit mode
set wildmenu              " This is used with wildmode(full) to cycle options

"Longer Set options
set cscopequickfix=s-,c-,d-,i-,t-,e-,g-,f-   " useful for cscope in quickfix
set listchars=tab:>-,trail:-                 " prefix tabs with a > and trails with -
set tags+=./.tags;/,./tags;/                 " set ctags
set whichwrap+=<,>,[,],h,l,~                 " arrow keys can wrap in normal and insert modes
set wildmode=list:longest,full               " list all options, match to the longest

set helpfile=\$VIMRUNTIME/doc/help.txt
set guifont=Courier\ 10\ Pitch\ 10
set path+=.,..,../..,../../..,../../../..,/usr/include

" Suffixes that get lower priority when doing tab completion for filenames.
" These are files I am not likely to want to edit or read.
set suffixes=.bak,~,.swp,.o,.info,.aux,.log,.dvi,.bbl,.blg,.brf,.cb,.ind,.idx,.ilg,.inx,.out,.toc,.class

"Disabled options
"set list                    " Make tabs and trails explicit
"set noswapfile              " this guy is really annoyoing sometimes
"set wrapmargin=80           " When pasteing, use this, because textwidth becomes 0
                             " wrapmargin inserts breaks if you exceed its value
"set cscopeprg=~/bin/cscope  "set cscope bin path

"Set colorscheme.  This is a black background colorscheme
colorscheme darkblue

" viminfo options
" read/write a .viminfo file, don't store more than
" 50 lines of registers
set viminfo='20,\"50

"Set variables for plugins to use

"vimspell variables
"don't automatically spell check!
let spell_auto_type=""

"taglist.vim settings
if exists("g:useNinjaTagList") && g:useNinjaTagList == 1
  set updatetime=1000 "interval to update list window

  let Tlist_Auto_Open=1 "Auto open the list window
  let Tlist_Compact_Format=1
  let Tlist_Ctags_Cmd = g:ApolloRoot . '/bin/ctags' "Ctags binary to use
  let Tlist_Enable_Fold_Column=0 "Turn off the fold column for list window
  let Tlist_Exit_OnlyWindow=1 "Exit if list is only window
  let Tlist_File_Fold_Auto_Close=1
  let Tlist_Show_Menu=1 "Show tag menu in gvim
  let Tlist_Use_Right_Window = 1 "put list window on the rigth

  "maps to close, and open list window
  map <silent> <Leader>tc :TlistClose<CR>
  map <silent> <Leader>to :TlistOpen<CR>
endif

" LargeFile.vim settings
" don't run syntax and other expensive things on files larger than NUM megs
let g:LargeFile = 100

"ag.vim settings

"Turn on filetype plugins to automagically
"Grab commands for particular filetypes.
"Grabbed from \$VIM/ftplugin
filetype plugin on
filetype indent on

"Turn on syntax highlighting
syntax on

"mappings
"normal mode maps
"Map \pe to editing the file in perforce
nmap <Leader>pe :!PWD=\`/apollo/env/envImprovement/bin/realdir %\` p4 edit \`basename %\`<CR>

"Map \e to edit a file from the directory of the current buffer
if has("unix")
  nmap <Leader>e :e <C-R>=expand("%:p:h") . "/"<CR>
else
  nmap <Leader>,e :e <C-R>=expand("%:p:h") . "\\"<CR>
endif

"map commands
"re-map rcsvers funtions
map <F5> <Leader>rlog  " Display log buffer selector
map <F6> <Leader>older " diff with an older version of the file
map <F7> <Leader>newer " diff with a newer version of the file
"firefox like tab switching
map [6^ :tabnext<CR> " Switch to the next tab
map [5^ :tabprev<CR> " Switch to the prev tab

"Functions
fu! CscopeAdd() " Add Cscope database named .cscope.out
    let dir = getcwd()
    let savedir = getcwd()
    wh (dir != '/')
        let scopefile = dir . '/' . ".cscope.out"
        if filereadable(scopefile)
            exe "cs add " scopefile
            exe "cd " savedir
            return dir
        en
        cd ..
        let dir = getcwd()
    endw
    exe "cd " savedir
endf

"Adding mail as a spell checked type, only if in 7.0 >
if (v:version >= 700)
  au FileType mail set spell
endif

"When editing a file, make screen display the name of the file you are editing
"Enabled by default. Either unlet variable or set to 0 in your .vimrc to disable.
let g:EnvImprovement_SetTitleEnabled = 1
function! SetTitle()
  if exists("g:EnvImprovement_SetTitleEnabled") && g:EnvImprovement_SetTitleEnabled && \$TERM =~ "^screen"
    let l:title = 'vi: ' . expand('%:t')

    if (l:title != 'vi: __Tag_List__')
      let l:truncTitle = strpart(l:title, 0, 15)
      silent exe '!echo -e -n "\033k' . l:truncTitle . '\033\\"'
    endif
  endif
endfunction

" Run it every time we change buffers
autocmd BufEnter,BufFilePost * call SetTitle()

"Automatically delete trailing whitespace on write for specified filetypes
" grab the file list from the variable g:EnvImprovement_DeleteWsFiletypes
" the variable should be of type list
function! DeleteTrailingWhitespace()
    let l:EnvImprovement_SaveCursor = getpos('.')
    %s/\s\+$//e
    call setpos('.', l:EnvImprovement_SaveCursor)
endfunction

if exists("g:EnvImprovement_DeleteWsFiletypes") && !empty(g:EnvImprovement_DeleteWsFiletypes)
    let filetypeString = join(g:EnvImprovement_DeleteWsFiletypes, ',')
    execute 'autocmd FileType ' . filetypeString  . ' autocmd BufWritePre <buffer> :call DeleteTrailingWhitespace()'
endif

:" Map Ctrl-A -> Start of line, Ctrl-E -> End of line
:map <C-a> <Home>
:map <C-e> <End>
EOL

echo "Setup complete! Please restart your terminal."
