require 'torch'

-- construct an object to deal with the mapping
local mapper = torch.class('Mapper')
--alphabet:a-z、$ tocken:0-26
function mapper:__init(dictPath)
    assert(paths.filep(dictPath), dictPath ..' not found')

    self.alphabet2token = {}
    self.token2alphabet = {}

    -- make maps
    local cnt = 0
    for line in io.lines(dictPath) do
        self.alphabet2token[line] = cnt
        self.token2alphabet[cnt] = line
        cnt = cnt + 1
    end
end
--io.lines():返回一个迭代函数,每次调用将获得文件中的一行内容,当到文件尾时，将返回nil,但不关闭文件
